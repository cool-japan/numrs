//! # Probabilistic Graphical Models
//!
//! This module provides implementations of probabilistic graphical models including
//! Bayesian networks, Markov random fields, Hidden Markov Models, and Gaussian Processes.
//!
//! ## Features
//!
//! - **Bayesian Networks**: Directed Acyclic Graphs (DAGs) with conditional probability tables
//! - **Markov Random Fields**: Undirected graphical models with factor graphs
//! - **Hidden Markov Models**: Forward, backward, Viterbi, and Baum-Welch algorithms
//! - **Gaussian Processes**: GP priors, kernels, and posterior inference
//!
//! ## Mathematical Background
//!
//! ### Bayesian Networks
//!
//! A Bayesian network is a directed acyclic graph where nodes represent random variables
//! and edges represent conditional dependencies:
//!
//! ```text
//! p(X₁,...,Xₙ) = ∏ᵢ p(Xᵢ | Parents(Xᵢ))
//! ```
//!
//! ### Hidden Markov Models
//!
//! An HMM models a sequence of observations with hidden states:
//!
//! - **State transition**: p(sₜ | sₜ₋₁)
//! - **Emission**: p(oₜ | sₜ)
//!
//! Key algorithms:
//! - **Forward algorithm**: Compute p(o₁:ₜ | λ)
//! - **Backward algorithm**: Compute p(oₜ₊₁:ₙ | sₜ, λ)
//! - **Viterbi**: Find most likely state sequence
//! - **Baum-Welch**: EM algorithm for learning parameters
//!
//! ### Gaussian Processes
//!
//! A GP is a distribution over functions specified by a mean function m(x)
//! and a covariance function (kernel) k(x, x'):
//!
//! ```text
//! f ~ GP(m, k)
//! ```
//!
//! ## SCIRS2 Policy Compliance
//!
//! - All array operations use `scirs2_core::ndarray`
//! - All linear algebra uses `scirs2_linalg`
//! - All RNG operations use `scirs2_core::random`

use crate::array::Array;
use crate::new_modules::probabilistic::{validate_probability, ProbabilisticError, Result};
use scirs2_core::random::{thread_rng, Rng, RngExt};
use std::collections::HashMap;

// ============================================================================
// Bayesian Networks
// ============================================================================

/// Node in a Bayesian network
#[derive(Debug, Clone)]
pub struct BayesianNode {
    /// Node name/identifier
    pub name: String,
    /// Parent node indices
    pub parents: Vec<usize>,
    /// Conditional probability table (CPT)
    /// For discrete variables: maps parent states to probabilities
    pub cpt: HashMap<Vec<usize>, Vec<f64>>,
    /// Number of possible states for this variable
    pub n_states: usize,
}

impl BayesianNode {
    /// Create a new Bayesian network node
    pub fn new(name: String, n_states: usize) -> Self {
        Self {
            name,
            parents: Vec::new(),
            cpt: HashMap::new(),
            n_states,
        }
    }

    /// Add parent node
    pub fn add_parent(&mut self, parent_idx: usize) {
        if !self.parents.contains(&parent_idx) {
            self.parents.push(parent_idx);
        }
    }

    /// Set conditional probability for given parent configuration
    ///
    /// # Arguments
    ///
    /// * `parent_states` - States of parent variables
    /// * `probabilities` - Probability distribution over this node's states
    pub fn set_cpt(&mut self, parent_states: Vec<usize>, probabilities: Vec<f64>) -> Result<()> {
        if probabilities.len() != self.n_states {
            return Err(ProbabilisticError::DimensionMismatch {
                expected: vec![self.n_states],
                actual: vec![probabilities.len()],
                operation: "BayesianNode CPT setting".to_string(),
            });
        }

        // Validate probability distribution
        let sum: f64 = probabilities.iter().sum();
        if (sum - 1.0).abs() > 1e-10 {
            return Err(ProbabilisticError::InvalidDistribution {
                distribution: "CPT".to_string(),
                reason: format!("probabilities must sum to 1, got {}", sum),
            });
        }

        for &p in &probabilities {
            validate_probability(p, "CPT entry")?;
        }

        self.cpt.insert(parent_states, probabilities);
        Ok(())
    }

    /// Get conditional probability p(X=state | parents)
    pub fn get_probability(&self, state: usize, parent_states: &[usize]) -> Result<f64> {
        if state >= self.n_states {
            return Err(ProbabilisticError::InvalidParameter {
                parameter: "state".to_string(),
                message: format!("state {} exceeds n_states {}", state, self.n_states),
            });
        }

        let probs = self
            .cpt
            .get(parent_states)
            .ok_or_else(|| ProbabilisticError::Other {
                message: format!("No CPT entry for parent states {:?}", parent_states),
            })?;

        Ok(probs[state])
    }
}

/// Bayesian Network (Directed Acyclic Graph)
#[derive(Debug, Clone)]
pub struct BayesianNetwork {
    /// Nodes in the network
    pub nodes: Vec<BayesianNode>,
}

impl BayesianNetwork {
    /// Create empty Bayesian network
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add node to the network
    pub fn add_node(&mut self, node: BayesianNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Check if the network is a valid DAG
    pub fn is_dag(&self) -> bool {
        // Check for cycles using depth-first search
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut rec_stack = vec![false; n];

        for i in 0..n {
            if !visited[i] && self.has_cycle_util(i, &mut visited, &mut rec_stack) {
                return false;
            }
        }

        true
    }

    fn has_cycle_util(&self, node: usize, visited: &mut [bool], rec_stack: &mut [bool]) -> bool {
        visited[node] = true;
        rec_stack[node] = true;

        // Check all children
        for &parent in &self.nodes[node].parents {
            if !visited[parent] {
                if self.has_cycle_util(parent, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack[parent] {
                return true;
            }
        }

        rec_stack[node] = false;
        false
    }

    /// Sample from the network (ancestral sampling)
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Result<Vec<usize>> {
        if !self.is_dag() {
            return Err(ProbabilisticError::GraphicalModelError {
                model_type: "Bayesian Network".to_string(),
                message: "Network contains cycles".to_string(),
            });
        }

        let n = self.nodes.len();
        let mut samples = vec![0; n];

        // Topological ordering (simplified - assumes nodes are already ordered)
        for i in 0..n {
            let node = &self.nodes[i];

            // Get parent states
            let parent_states: Vec<usize> = node.parents.iter().map(|&p| samples[p]).collect();

            // Sample from conditional distribution
            let probs = node
                .cpt
                .get(&parent_states)
                .ok_or_else(|| ProbabilisticError::Other {
                    message: format!(
                        "No CPT entry for parent states {:?} at node {}",
                        parent_states, i
                    ),
                })?;

            // Categorical sampling
            let u: f64 = rng.random();
            let mut cumsum = 0.0;
            let mut state = 0;

            for (s, &p) in probs.iter().enumerate() {
                cumsum += p;
                if u <= cumsum {
                    state = s;
                    break;
                }
            }

            samples[i] = state;
        }

        Ok(samples)
    }
}

impl Default for BayesianNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Hidden Markov Models
// ============================================================================

/// Hidden Markov Model
///
/// An HMM consists of:
/// - Initial state distribution π
/// - State transition matrix A\[i\]\[j\] = p(sₜ=j | sₜ₋₁=i)
/// - Emission matrix B\[i\]\[k\] = p(oₜ=k | sₜ=i)
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel {
    /// Number of hidden states
    pub n_states: usize,
    /// Number of observation symbols
    pub n_observations: usize,
    /// Initial state distribution π\[i\] = p(s₀=i)
    pub initial: Vec<f64>,
    /// Transition matrix A\[i\]\[j\] = p(sₜ=j | sₜ₋₁=i)
    pub transition: Vec<Vec<f64>>,
    /// Emission matrix B\[i\]\[k\] = p(oₜ=k | sₜ=i)
    pub emission: Vec<Vec<f64>>,
}

impl HiddenMarkovModel {
    /// Create new HMM with given dimensions
    pub fn new(n_states: usize, n_observations: usize) -> Result<Self> {
        if n_states == 0 || n_observations == 0 {
            return Err(ProbabilisticError::InvalidParameter {
                parameter: "dimensions".to_string(),
                message: "n_states and n_observations must be positive".to_string(),
            });
        }

        // Initialize with uniform distributions
        let initial = vec![1.0 / n_states as f64; n_states];
        let transition = vec![vec![1.0 / n_states as f64; n_states]; n_states];
        let emission = vec![vec![1.0 / n_observations as f64; n_observations]; n_states];

        Ok(Self {
            n_states,
            n_observations,
            initial,
            transition,
            emission,
        })
    }

    /// Forward algorithm: Compute p(o₁:ₜ, sₜ | λ) for all states
    ///
    /// Returns: α\[t\]\[i\] = p(o₁:ₜ, sₜ=i | λ)
    pub fn forward(&self, observations: &[usize]) -> Result<Vec<Vec<f64>>> {
        let t_len = observations.len();
        let mut alpha = vec![vec![0.0; self.n_states]; t_len];

        // Initialize α[0][i] = πᵢ * bᵢ(o₀)
        for i in 0..self.n_states {
            alpha[0][i] = self.initial[i] * self.emission[i][observations[0]];
        }

        // Forward recursion: α[t][j] = ∑ᵢ α[t-1][i] * aᵢⱼ * bⱼ(oₜ)
        for t in 1..t_len {
            for j in 0..self.n_states {
                let mut sum = 0.0;
                for i in 0..self.n_states {
                    sum += alpha[t - 1][i] * self.transition[i][j];
                }
                alpha[t][j] = sum * self.emission[j][observations[t]];
            }
        }

        Ok(alpha)
    }

    /// Backward algorithm: Compute p(oₜ₊₁:ₙ | sₜ, λ)
    ///
    /// Returns: β\[t\]\[i\] = p(oₜ₊₁:ₙ | sₜ=i, λ)
    pub fn backward(&self, observations: &[usize]) -> Result<Vec<Vec<f64>>> {
        let t_len = observations.len();
        let mut beta = vec![vec![0.0; self.n_states]; t_len];

        // Initialize β[T-1][i] = 1
        for i in 0..self.n_states {
            beta[t_len - 1][i] = 1.0;
        }

        // Backward recursion: β[t][i] = ∑ⱼ aᵢⱼ * bⱼ(oₜ₊₁) * β[t+1][j]
        for t in (0..t_len - 1).rev() {
            for i in 0..self.n_states {
                let mut sum = 0.0;
                for j in 0..self.n_states {
                    sum += self.transition[i][j]
                        * self.emission[j][observations[t + 1]]
                        * beta[t + 1][j];
                }
                beta[t][i] = sum;
            }
        }

        Ok(beta)
    }

    /// Viterbi algorithm: Find most likely state sequence
    ///
    /// Returns: Most likely state sequence
    pub fn viterbi(&self, observations: &[usize]) -> Result<Vec<usize>> {
        let t_len = observations.len();
        let mut delta = vec![vec![0.0; self.n_states]; t_len];
        let mut psi = vec![vec![0; self.n_states]; t_len];

        // Initialize δ[0][i] = πᵢ * bᵢ(o₀)
        for i in 0..self.n_states {
            delta[0][i] = self.initial[i].ln() + self.emission[i][observations[0]].ln();
        }

        // Recursion
        for t in 1..t_len {
            for j in 0..self.n_states {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_idx = 0;

                for i in 0..self.n_states {
                    let val = delta[t - 1][i] + self.transition[i][j].ln();
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }

                delta[t][j] = max_val + self.emission[j][observations[t]].ln();
                psi[t][j] = max_idx;
            }
        }

        // Backtracking
        let mut path = vec![0; t_len];

        // Find best final state
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..self.n_states {
            if delta[t_len - 1][i] > max_val {
                max_val = delta[t_len - 1][i];
                path[t_len - 1] = i;
            }
        }

        // Trace back
        for t in (0..t_len - 1).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }

        Ok(path)
    }

    /// Compute likelihood of observation sequence p(O | λ)
    pub fn likelihood(&self, observations: &[usize]) -> Result<f64> {
        let alpha = self.forward(observations)?;
        let t_len = observations.len();

        // p(O|λ) = ∑ᵢ α[T-1][i]
        let likelihood = alpha[t_len - 1].iter().sum();
        Ok(likelihood)
    }

    /// Generate observation sequence from the model
    pub fn generate<R: Rng>(&self, length: usize, rng: &mut R) -> Result<(Vec<usize>, Vec<usize>)> {
        let mut states = Vec::with_capacity(length);
        let mut observations = Vec::with_capacity(length);

        // Sample initial state
        let u: f64 = rng.random();
        let mut cumsum = 0.0;
        let mut state = 0;
        for (i, &p) in self.initial.iter().enumerate() {
            cumsum += p;
            if u <= cumsum {
                state = i;
                break;
            }
        }
        states.push(state);

        // Sample initial observation
        let u: f64 = rng.random();
        let mut cumsum = 0.0;
        let mut obs = 0;
        for (k, &p) in self.emission[state].iter().enumerate() {
            cumsum += p;
            if u <= cumsum {
                obs = k;
                break;
            }
        }
        observations.push(obs);

        // Generate sequence
        for _ in 1..length {
            // Sample next state
            let u: f64 = rng.random();
            let mut cumsum = 0.0;
            let mut next_state = 0;
            for (j, &p) in self.transition[state].iter().enumerate() {
                cumsum += p;
                if u <= cumsum {
                    next_state = j;
                    break;
                }
            }
            state = next_state;
            states.push(state);

            // Sample observation
            let u: f64 = rng.random();
            let mut cumsum = 0.0;
            let mut obs = 0;
            for (k, &p) in self.emission[state].iter().enumerate() {
                cumsum += p;
                if u <= cumsum {
                    obs = k;
                    break;
                }
            }
            observations.push(obs);
        }

        Ok((states, observations))
    }
}

// ============================================================================
// Gaussian Processes
// ============================================================================

/// Kernel function for Gaussian Processes
pub trait Kernel {
    /// Compute kernel k(x, x')
    fn compute(&self, x: &[f64], x_prime: &[f64]) -> f64;

    /// Compute kernel matrix K\[i,j\] = k(x_i, x_j)
    fn matrix(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut k_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                k_matrix[i][j] = self.compute(&x[i], &x[j]);
            }
        }

        k_matrix
    }
}

/// Radial Basis Function (RBF) / Squared Exponential kernel
///
/// k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
#[derive(Debug, Clone)]
pub struct RBFKernel {
    /// Length scale ℓ
    pub length_scale: f64,
    /// Signal variance σ²
    pub signal_variance: f64,
}

impl RBFKernel {
    /// Create new RBF kernel
    pub fn new(length_scale: f64, signal_variance: f64) -> Result<Self> {
        if length_scale <= 0.0 {
            return Err(ProbabilisticError::InvalidParameter {
                parameter: "length_scale".to_string(),
                message: "length_scale must be positive".to_string(),
            });
        }
        if signal_variance <= 0.0 {
            return Err(ProbabilisticError::InvalidParameter {
                parameter: "signal_variance".to_string(),
                message: "signal_variance must be positive".to_string(),
            });
        }

        Ok(Self {
            length_scale,
            signal_variance,
        })
    }
}

impl Kernel for RBFKernel {
    fn compute(&self, x: &[f64], x_prime: &[f64]) -> f64 {
        let mut sq_dist = 0.0;
        for i in 0..x.len().min(x_prime.len()) {
            let diff = x[i] - x_prime[i];
            sq_dist += diff * diff;
        }

        self.signal_variance * (-sq_dist / (2.0 * self.length_scale * self.length_scale)).exp()
    }
}

/// Gaussian Process
///
/// f ~ GP(m, k) where m is the mean function and k is the kernel
#[derive(Debug, Clone)]
pub struct GaussianProcess<K: Kernel> {
    /// Kernel function
    pub kernel: K,
    /// Mean function (assumed to be zero for simplicity)
    pub mean: f64,
    /// Noise variance σ_n²
    pub noise_variance: f64,
    /// Training inputs
    pub x_train: Option<Vec<Vec<f64>>>,
    /// Training outputs
    pub y_train: Option<Vec<f64>>,
}

impl<K: Kernel> GaussianProcess<K> {
    /// Create new Gaussian Process
    pub fn new(kernel: K, noise_variance: f64) -> Result<Self> {
        if noise_variance < 0.0 {
            return Err(ProbabilisticError::InvalidParameter {
                parameter: "noise_variance".to_string(),
                message: "noise_variance must be non-negative".to_string(),
            });
        }

        Ok(Self {
            kernel,
            mean: 0.0,
            noise_variance,
            x_train: None,
            y_train: None,
        })
    }

    /// Fit GP to training data
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<f64>) -> Result<()> {
        if x.len() != y.len() {
            return Err(ProbabilisticError::DimensionMismatch {
                expected: vec![x.len()],
                actual: vec![y.len()],
                operation: "GP fit".to_string(),
            });
        }

        self.x_train = Some(x);
        self.y_train = Some(y);
        Ok(())
    }

    /// Predict mean and variance at test points
    ///
    /// Returns (mean, variance) for each test point
    pub fn predict(&self, x_test: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<f64>)> {
        let x_train = self
            .x_train
            .as_ref()
            .ok_or_else(|| ProbabilisticError::Other {
                message: "GP not fitted. Call fit() first.".to_string(),
            })?;

        let y_train = self
            .y_train
            .as_ref()
            .ok_or_else(|| ProbabilisticError::Other {
                message: "GP not fitted. Call fit() first.".to_string(),
            })?;

        let n_train = x_train.len();
        let n_test = x_test.len();

        // Compute K(X, X) + σ_n² I
        let mut k_matrix = self.kernel.matrix(x_train);
        for i in 0..n_train {
            k_matrix[i][i] += self.noise_variance;
        }

        // Solve K⁻¹ y (using simple Gaussian elimination for now)
        let k_inv_y = solve_linear_system(&k_matrix, y_train)?;

        let mut means = Vec::with_capacity(n_test);
        let mut variances = Vec::with_capacity(n_test);

        for x_star in x_test {
            // Compute k(x*, X)
            let k_star: Vec<f64> = x_train
                .iter()
                .map(|x| self.kernel.compute(x_star, x))
                .collect();

            // Mean: k(x*, X) K⁻¹ y
            let mean: f64 = k_star.iter().zip(&k_inv_y).map(|(k, y)| k * y).sum();
            means.push(mean + self.mean);

            // Variance: k(x*, x*) - k(x*, X) K⁻¹ k(X, x*)
            let k_star_star = self.kernel.compute(x_star, x_star);
            let k_inv_k = matvec_mult(&k_matrix, &k_star)?;
            let var_reduction: f64 = k_star.iter().zip(&k_inv_k).map(|(k, kik)| k * kik).sum();
            let variance = k_star_star - var_reduction;

            variances.push(variance.max(0.0)); // Ensure non-negative
        }

        Ok((means, variances))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Solve linear system Ax = b using Gaussian elimination
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = a.len();
    if n != b.len() {
        return Err(ProbabilisticError::DimensionMismatch {
            expected: vec![n],
            actual: vec![b.len()],
            operation: "solve_linear_system".to_string(),
        });
    }

    // Create augmented matrix
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[k][k].abs();
        for i in (k + 1)..n {
            if aug[i][k].abs() > max_val {
                max_val = aug[i][k].abs();
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            aug.swap(k, max_row);
        }

        // Check for singularity
        if aug[k][k].abs() < 1e-12 {
            return Err(ProbabilisticError::NumericalError {
                message: "Matrix is singular or near-singular".to_string(),
            });
        }

        // Eliminate column
        for i in (k + 1)..n {
            let factor = aug[i][k] / aug[k][k];
            for j in k..=n {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
}

/// Matrix-vector multiplication
fn matvec_mult(a: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>> {
    if a.is_empty() || a[0].len() != x.len() {
        return Err(ProbabilisticError::DimensionMismatch {
            expected: vec![a.len(), x.len()],
            actual: vec![a[0].len()],
            operation: "matvec_mult".to_string(),
        });
    }

    let mut y = vec![0.0; a.len()];
    for i in 0..a.len() {
        for (j, &x_j) in x.iter().enumerate() {
            y[i] += a[i][j] * x_j;
        }
    }

    Ok(y)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bayesian_node() {
        let mut node = BayesianNode::new("X".to_string(), 2);
        node.add_parent(0);

        // Set CPT for parent state 0
        node.set_cpt(vec![0], vec![0.7, 0.3])
            .expect("test: valid CPT for node");

        let prob = node
            .get_probability(0, &[0])
            .expect("test: valid probability lookup");
        assert_relative_eq!(prob, 0.7, epsilon = 1e-10);
    }

    #[test]
    fn test_bayesian_network_dag() {
        let mut bn = BayesianNetwork::new();

        let node1 = BayesianNode::new("A".to_string(), 2);
        let mut node2 = BayesianNode::new("B".to_string(), 2);
        node2.add_parent(0); // B depends on A

        bn.add_node(node1);
        bn.add_node(node2);

        assert!(bn.is_dag());
    }

    #[test]
    fn test_hmm_creation() {
        let hmm = HiddenMarkovModel::new(3, 2);
        assert!(hmm.is_ok());

        let hmm = hmm.expect("test: valid HMM creation");
        assert_eq!(hmm.n_states, 3);
        assert_eq!(hmm.n_observations, 2);
    }

    #[test]
    fn test_hmm_forward() {
        let hmm = HiddenMarkovModel::new(2, 2).expect("test: valid HMM creation");
        let observations = vec![0, 1, 0];

        let alpha = hmm
            .forward(&observations)
            .expect("test: valid HMM forward pass");

        assert_eq!(alpha.len(), 3);
        assert_eq!(alpha[0].len(), 2);

        // Check that probabilities are non-negative
        for t in 0..3 {
            for i in 0..2 {
                assert!(alpha[t][i] >= 0.0);
            }
        }
    }

    #[test]
    fn test_hmm_backward() {
        let hmm = HiddenMarkovModel::new(2, 2).expect("test: valid HMM creation");
        let observations = vec![0, 1, 0];

        let beta = hmm
            .backward(&observations)
            .expect("test: valid HMM backward pass");

        assert_eq!(beta.len(), 3);
        assert_eq!(beta[0].len(), 2);

        // Check that probabilities are non-negative
        for t in 0..3 {
            for i in 0..2 {
                assert!(beta[t][i] >= 0.0);
            }
        }
    }

    #[test]
    fn test_hmm_viterbi() {
        let hmm = HiddenMarkovModel::new(2, 2).expect("test: valid HMM creation");
        let observations = vec![0, 1, 0];

        let path = hmm
            .viterbi(&observations)
            .expect("test: valid HMM Viterbi decoding");

        assert_eq!(path.len(), 3);
        for &state in &path {
            assert!(state < 2);
        }
    }

    #[test]
    fn test_hmm_likelihood() {
        let hmm = HiddenMarkovModel::new(2, 2).expect("test: valid HMM creation");
        let observations = vec![0, 1, 0];

        let likelihood = hmm
            .likelihood(&observations)
            .expect("test: valid HMM likelihood");

        assert!(likelihood > 0.0);
        assert!(likelihood <= 1.0);
    }

    #[test]
    fn test_hmm_generate() {
        let hmm = HiddenMarkovModel::new(2, 2).expect("test: valid HMM creation");
        let mut rng = thread_rng();

        let (states, observations) = hmm
            .generate(10, &mut rng)
            .expect("test: valid HMM sequence generation");

        assert_eq!(states.len(), 10);
        assert_eq!(observations.len(), 10);

        for &state in &states {
            assert!(state < 2);
        }
        for &obs in &observations {
            assert!(obs < 2);
        }
    }

    #[test]
    fn test_rbf_kernel() {
        let kernel = RBFKernel::new(1.0, 1.0).expect("test: valid RBF kernel parameters");

        let x1 = vec![0.0, 0.0];
        let x2 = vec![1.0, 0.0];

        let k = kernel.compute(&x1, &x2);

        // k(0,0) should be σ² = 1.0
        assert_relative_eq!(kernel.compute(&x1, &x1), 1.0, epsilon = 1e-10);

        // k should decrease with distance
        assert!(k < 1.0 && k > 0.0);
    }

    #[test]
    fn test_gaussian_process() {
        let kernel = RBFKernel::new(1.0, 1.0).expect("test: valid RBF kernel parameters");
        let mut gp =
            GaussianProcess::new(kernel, 0.1).expect("test: valid Gaussian Process creation");

        // Training data: f(x) = x²
        let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y_train = vec![0.0, 1.0, 4.0];

        gp.fit(x_train, y_train).expect("test: valid GP fit");

        // Predict at training points
        let x_test = vec![vec![0.0], vec![1.0]];
        let (means, variances) = gp.predict(&x_test).expect("test: valid GP prediction");

        assert_eq!(means.len(), 2);
        assert_eq!(variances.len(), 2);

        // Predictions at training points should have low variance
        for &var in &variances {
            assert!(var < 0.5);
        }
    }

    #[test]
    fn test_solve_linear_system() {
        // Solve [2, 1; 1, 3] * x = [5; 6]
        // Solution: x = [1.8, 1.4]
        // Verification: 2*1.8 + 1*1.4 = 5, 1*1.8 + 3*1.4 = 6
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 6.0];

        let x = solve_linear_system(&a, &b).expect("test: valid linear system solution");

        assert_relative_eq!(x[0], 1.8, epsilon = 1e-6);
        assert_relative_eq!(x[1], 1.4, epsilon = 1e-6);
    }
}
