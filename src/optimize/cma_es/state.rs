//! Internal CMA-ES optimizer state.

use crate::error::{NumRs2Error, Result};

use super::config::{compute_default_weights, compute_mu_eff, CMAESConfig};
use super::eigen::symmetric_eigendecomposition;
use super::types::{RngSource, TerminationReason};

/// Internal CMA-ES optimizer state.
pub(crate) struct CmaEsState {
    /// Problem dimension
    pub(crate) n: usize,
    /// Mean of the distribution
    pub(crate) mean: Vec<f64>,
    /// Step-size
    pub(crate) sigma: f64,
    /// Covariance matrix (n x n, stored row-major)
    pub(crate) covariance: Vec<f64>,
    /// Eigenvalues of C
    pub(crate) eigenvalues: Vec<f64>,
    /// Eigenvectors of C (columns = eigenvectors, stored row-major)
    pub(crate) eigenvectors: Vec<f64>,
    /// sqrt of eigenvalues for sampling
    pub(crate) sqrt_eigenvalues: Vec<f64>,
    /// Evolution path for covariance (p_c)
    pub(crate) pc: Vec<f64>,
    /// Evolution path for step-size (p_sigma)
    pub(crate) ps: Vec<f64>,
    /// Population size lambda
    pub(crate) lambda: usize,
    /// Number of parents mu
    pub(crate) mu: usize,
    /// Recombination weights
    pub(crate) weights: Vec<f64>,
    /// Variance effective selection mass
    pub(crate) mu_eff: f64,
    /// Learning rate for rank-1 update
    pub(crate) c1: f64,
    /// Learning rate for rank-mu update
    pub(crate) cmu: f64,
    /// Learning rate for p_c cumulation
    pub(crate) cc: f64,
    /// Learning rate for p_sigma cumulation
    pub(crate) cs: f64,
    /// Damping for step-size
    pub(crate) damps: f64,
    /// Expected length of N(0,I) vector
    pub(crate) chi_n: f64,
    /// Generation counter
    pub(crate) generation: usize,
    /// Function evaluation counter
    pub(crate) function_evaluations: usize,
    /// Best solution found so far
    pub(crate) best_x: Vec<f64>,
    /// Best function value found so far
    pub(crate) best_f: f64,
    /// Convergence history
    pub(crate) history: Vec<f64>,
    /// Whether eigendecomposition is up to date
    pub(crate) eigen_updated: bool,
    /// Counter for eigendecomposition updates
    pub(crate) eigen_update_counter: usize,
    /// RNG source
    pub(crate) rng: RngSource,
}

impl CmaEsState {
    /// Initialize the CMA-ES state from a configuration and starting point.
    pub(crate) fn new(x0: &[f64], config: &CMAESConfig) -> Result<Self> {
        let n = x0.len();
        if n == 0 {
            return Err(NumRs2Error::InvalidInput(
                "Initial point must have at least one dimension".to_string(),
            ));
        }
        if config.sigma0 <= 0.0 {
            return Err(NumRs2Error::InvalidInput(
                "Initial step-size sigma0 must be positive".to_string(),
            ));
        }

        let lambda = config.effective_lambda(n);
        let mu = lambda / 2;
        if mu == 0 {
            return Err(NumRs2Error::InvalidInput(
                "Population size too small: mu would be 0".to_string(),
            ));
        }

        // Validate bounds if provided
        if let Some(ref bounds) = config.bounds {
            if bounds.len() != n {
                return Err(NumRs2Error::InvalidInput(format!(
                    "Bounds length {} does not match dimension {}",
                    bounds.len(),
                    n
                )));
            }
            for (i, &(lo, hi)) in bounds.iter().enumerate() {
                if lo >= hi {
                    return Err(NumRs2Error::InvalidInput(format!(
                        "Invalid bounds for dimension {}: lower ({}) >= upper ({})",
                        i, lo, hi
                    )));
                }
            }
        }

        // Compute recombination weights (log-linear)
        let weights = match config.weights {
            Some(ref w) => w.clone(),
            None => compute_default_weights(mu, lambda),
        };

        // Effective selection mass
        let mu_eff = compute_mu_eff(&weights, mu);

        let n_f = n as f64;

        // Learning rate for cumulation for step-size control
        let cs = config
            .cs
            .unwrap_or_else(|| (mu_eff + 2.0) / (n_f + mu_eff + 5.0));

        // Damping for step-size
        let damps = config.damps.unwrap_or_else(|| {
            1.0 + 2.0 * (((mu_eff - 1.0) / (n_f + 1.0)).sqrt() - 1.0).max(0.0) + cs
        });

        // Learning rate for cumulation of C
        let cc = config
            .cc
            .unwrap_or_else(|| (4.0 + mu_eff / n_f) / (n_f + 4.0 + 2.0 * mu_eff / n_f));

        // Learning rate for rank-1 update
        let c1 = config
            .c1
            .unwrap_or_else(|| 2.0 / ((n_f + 1.3).powi(2) + mu_eff));

        // Learning rate for rank-mu update
        let cmu = config.cmu.unwrap_or_else(|| {
            let a = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n_f + 2.0).powi(2) + mu_eff);
            a.min(1.0 - c1)
        });

        // Expected length of a N(0,I) distributed vector
        let chi_n = n_f.sqrt() * (1.0 - 1.0 / (4.0 * n_f) + 1.0 / (21.0 * n_f * n_f));

        // Initialize covariance as identity
        let mut covariance = vec![0.0; n * n];
        for i in 0..n {
            covariance[i * n + i] = 1.0;
        }

        // Eigenvalues and eigenvectors of identity
        let eigenvalues = vec![1.0; n];
        let sqrt_eigenvalues = vec![1.0; n];
        let mut eigenvectors = vec![0.0; n * n];
        for i in 0..n {
            eigenvectors[i * n + i] = 1.0;
        }

        let rng = RngSource::create(config.seed);

        Ok(CmaEsState {
            n,
            mean: x0.to_vec(),
            sigma: config.sigma0,
            covariance,
            eigenvalues,
            eigenvectors,
            sqrt_eigenvalues,
            pc: vec![0.0; n],
            ps: vec![0.0; n],
            lambda,
            mu,
            weights,
            mu_eff,
            c1,
            cmu,
            cc,
            cs,
            damps,
            chi_n,
            generation: 0,
            function_evaluations: 0,
            best_x: x0.to_vec(),
            best_f: f64::INFINITY,
            history: Vec::new(),
            eigen_updated: true,
            eigen_update_counter: 0,
            rng,
        })
    }

    /// Sample lambda candidate solutions from N(m, sigma^2 C).
    ///
    /// Uses the eigendecomposition: x = m + sigma * B * D * z
    /// where B = eigenvectors, D = diag(sqrt(eigenvalues)), z ~ N(0, I).
    pub(crate) fn sample_population(&mut self) -> Vec<Vec<f64>> {
        let mut population = Vec::with_capacity(self.lambda);

        for _ in 0..self.lambda {
            // Sample z ~ N(0, I)
            let z: Vec<f64> = (0..self.n).map(|_| self.rng.sample_normal()).collect();

            // Transform: x = m + sigma * B * D * z
            let mut x = vec![0.0; self.n];
            for i in 0..self.n {
                let mut val = 0.0;
                for j in 0..self.n {
                    val += self.eigenvectors[i * self.n + j] * self.sqrt_eigenvalues[j] * z[j];
                }
                x[i] = self.mean[i] + self.sigma * val;
            }

            population.push(x);
        }

        population
    }

    /// Repair a candidate solution by clamping to bounds.
    pub(crate) fn repair_bounds(x: &mut [f64], bounds: &[(f64, f64)]) {
        for (xi, &(lo, hi)) in x.iter_mut().zip(bounds.iter()) {
            *xi = xi.clamp(lo, hi);
        }
    }

    /// Evaluate objective function with optional box constraint penalty.
    pub(crate) fn evaluate<F: Fn(&[f64]) -> f64>(
        &mut self,
        f: &F,
        x: &[f64],
        bounds: &Option<Vec<(f64, f64)>>,
        penalty_coeff: f64,
    ) -> f64 {
        self.function_evaluations += 1;
        let fval = f(x);

        // Add penalty for box constraint violations
        if let Some(ref b) = bounds {
            let mut penalty = 0.0;
            for (&xi, &(lo, hi)) in x.iter().zip(b.iter()) {
                if xi < lo {
                    let violation = lo - xi;
                    penalty += violation * violation;
                } else if xi > hi {
                    let violation = xi - hi;
                    penalty += violation * violation;
                }
            }
            fval + penalty_coeff * penalty
        } else {
            fval
        }
    }

    /// Update the mean (weighted recombination of the mu best solutions).
    ///
    /// Returns the old mean for step computation.
    pub(crate) fn update_mean(&mut self, sorted_pop: &[Vec<f64>]) -> Vec<f64> {
        let old_mean = self.mean.clone();
        for i in 0..self.n {
            let mut new_val = 0.0;
            for k in 0..self.mu {
                new_val += self.weights[k] * sorted_pop[k][i];
            }
            self.mean[i] = new_val;
        }
        old_mean
    }

    /// Compute the weighted step in the transformed space: y_w = (m_new - m_old) / sigma.
    pub(crate) fn compute_weighted_step(&self, old_mean: &[f64]) -> Vec<f64> {
        let inv_sigma = 1.0 / self.sigma;
        (0..self.n)
            .map(|i| (self.mean[i] - old_mean[i]) * inv_sigma)
            .collect()
    }

    /// Apply C^{-1/2} to a vector: C^{-1/2} v = B D^{-1} B^T v.
    fn apply_c_inv_sqrt(&self, v: &[f64]) -> Vec<f64> {
        let n = self.n;
        // First: u = B^T * v
        let mut u = vec![0.0; n];
        for j in 0..n {
            for i in 0..n {
                u[j] += self.eigenvectors[i * n + j] * v[i];
            }
        }

        // Second: scale by D^{-1}
        for j in 0..n {
            if self.sqrt_eigenvalues[j] > 1e-300 {
                u[j] /= self.sqrt_eigenvalues[j];
            }
        }

        // Third: result = B * u
        let mut result = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                result[i] += self.eigenvectors[i * n + j] * u[j];
            }
        }
        result
    }

    /// Update the evolution path for step-size (p_sigma) and adapt sigma via CSA.
    pub(crate) fn update_step_size(&mut self, y_w: &[f64]) {
        // Compute C^{-1/2} * y_w
        let c_inv_sqrt_y = self.apply_c_inv_sqrt(y_w);

        // Update p_sigma
        let cs_complement = (1.0 - self.cs).sqrt();
        let cs_factor = (self.cs * (2.0 - self.cs) * self.mu_eff).sqrt();

        for i in 0..self.n {
            self.ps[i] = cs_complement * self.ps[i] + cs_factor * c_inv_sqrt_y[i];
        }

        // Compute ||p_sigma||
        let ps_norm: f64 = self.ps.iter().map(|&v| v * v).sum::<f64>().sqrt();

        // Adapt sigma
        let ratio = ps_norm / self.chi_n - 1.0;
        self.sigma *= (self.cs / self.damps * ratio).exp();

        // Clamp sigma to prevent numerical issues
        self.sigma = self.sigma.clamp(1e-300, 1e100);
    }

    /// Update the covariance matrix using rank-1 and rank-mu updates.
    ///
    /// This is the core of CMA-ES: the covariance matrix learns the
    /// second-order structure of the objective function landscape.
    pub(crate) fn update_covariance(
        &mut self,
        y_w: &[f64],
        sorted_pop: &[Vec<f64>],
        old_mean: &[f64],
    ) {
        let n = self.n;

        // Heaviside function for p_sigma stalling detection
        let ps_norm_sq: f64 = self.ps.iter().map(|&v| v * v).sum();
        let gen_factor = 2.0 * (self.generation as f64 + 1.0);
        let threshold = (1.0 - (1.0 - self.cs).powf(gen_factor)) * (n as f64 + 0.5);
        let h_sigma: f64 = if ps_norm_sq / threshold < (n as f64) + 4.0 * (n as f64).sqrt() {
            1.0
        } else {
            0.0
        };

        // Update pc (evolution path for covariance)
        let cc_complement = (1.0 - self.cc).sqrt();
        let cc_factor = h_sigma * (self.cc * (2.0 - self.cc) * self.mu_eff).sqrt();

        for i in 0..n {
            self.pc[i] = cc_complement * self.pc[i] + cc_factor * y_w[i];
        }

        // Delta for the h_sigma correction
        let delta_h = (1.0 - h_sigma) * self.cc * (2.0 - self.cc);

        // Old covariance factor
        let weight_sum: f64 = self.weights.iter().take(self.mu).sum();
        let c_old_factor = (1.0 + self.c1 * delta_h - self.c1 - self.cmu * weight_sum).max(0.0);

        // Rank-1 and rank-mu updates
        let inv_sigma = 1.0 / self.sigma;

        for i in 0..n {
            for j in 0..=i {
                // Old covariance contribution
                let mut new_val = c_old_factor * self.covariance[i * n + j];

                // Rank-1 update: c1 * pc * pc^T
                new_val += self.c1 * self.pc[i] * self.pc[j];

                // Rank-mu update: cmu * sum(w_k * y_k * y_k^T)
                let mut rank_mu_sum = 0.0;
                for k in 0..self.mu {
                    let y_k_i = (sorted_pop[k][i] - old_mean[i]) * inv_sigma;
                    let y_k_j = (sorted_pop[k][j] - old_mean[j]) * inv_sigma;
                    rank_mu_sum += self.weights[k] * y_k_i * y_k_j;
                }
                new_val += self.cmu * rank_mu_sum;

                self.covariance[i * n + j] = new_val;
                self.covariance[j * n + i] = new_val; // Symmetric
            }
        }

        self.eigen_updated = false;
    }

    /// Update eigendecomposition of the covariance matrix.
    ///
    /// Uses the Jacobi eigenvalue algorithm. This is the most expensive
    /// operation per generation; the update frequency follows Hansen's
    /// heuristic to amortize the cost.
    pub(crate) fn update_eigendecomposition(&mut self) -> Result<()> {
        let n = self.n;

        // Frequency of eigendecomposition update (heuristic from Hansen)
        let c1_plus_cmu = self.c1 + self.cmu;
        let update_freq = if c1_plus_cmu > 0.0 {
            ((n as f64) / (10.0 * c1_plus_cmu * (n as f64))).max(1.0) as usize
        } else {
            1
        };

        self.eigen_update_counter += 1;
        if self.eigen_updated || self.eigen_update_counter < update_freq {
            return Ok(());
        }
        self.eigen_update_counter = 0;

        // Symmetrize the covariance matrix (numerical safeguard)
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (self.covariance[i * n + j] + self.covariance[j * n + i]);
                self.covariance[i * n + j] = avg;
                self.covariance[j * n + i] = avg;
            }
        }

        // Perform eigendecomposition via Jacobi iteration
        let (eigenvalues, eigenvectors) = symmetric_eigendecomposition(&self.covariance, n)?;

        // Validate and store eigenvalues
        for (idx, ev) in eigenvalues.iter().enumerate() {
            if *ev < 0.0 {
                // Numerical drift: clamp to small positive value
                let clamped = ev.abs().max(1e-20);
                self.eigenvalues[idx] = clamped;
                self.sqrt_eigenvalues[idx] = clamped.sqrt();
            } else {
                self.eigenvalues[idx] = *ev;
                self.sqrt_eigenvalues[idx] = ev.sqrt();
            }
        }

        self.eigenvectors = eigenvectors;
        self.eigen_updated = true;

        Ok(())
    }

    /// Get the condition number of the covariance matrix.
    pub(crate) fn condition_number(&self) -> f64 {
        let max_ev = self
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_ev = self
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if min_ev > 0.0 {
            max_ev / min_ev
        } else {
            f64::INFINITY
        }
    }

    /// Check termination criteria and return the reason if terminated.
    pub(crate) fn check_termination(
        &self,
        config: &CMAESConfig,
        fitness_values: &[f64],
    ) -> Option<TerminationReason> {
        // Max generations
        if self.generation >= config.max_iter {
            return Some(TerminationReason::MaxGenerations);
        }

        // Function value tolerance: range of fitness in current generation is small
        if fitness_values.len() >= 2 && self.generation > 1 {
            let f_min = fitness_values.iter().cloned().fold(f64::INFINITY, f64::min);
            let f_max = fitness_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            if (f_max - f_min).abs() < config.ftol {
                return Some(TerminationReason::FunctionTolerance);
            }
        }

        // Parameter tolerance: all components of sigma * sqrt(diag(C)) are small
        if self.generation > 1 {
            let max_std = self
                .eigenvalues
                .iter()
                .map(|&ev| self.sigma * ev.sqrt())
                .fold(f64::NEG_INFINITY, f64::max);
            if max_std < config.xtol {
                return Some(TerminationReason::ParameterTolerance);
            }
        }

        // Condition number
        let cond = self.condition_number();
        if cond > config.max_condition_number {
            return Some(TerminationReason::ConditionNumber);
        }

        // Step-size diverged
        if self.sigma.is_nan() || self.sigma.is_infinite() || self.sigma < 1e-300 {
            return Some(TerminationReason::StepSizeDiverged);
        }

        // Eigenvalue degeneration
        if self.eigenvalues.iter().all(|&ev| ev < 1e-30) {
            return Some(TerminationReason::EigenvalueDegenerate);
        }

        None
    }
}
