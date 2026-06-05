//! Reinforcement Learning Agents
//!
//! This module provides various RL agents including tabular methods
//! (Q-Learning, SARSA) and deep RL methods (DQN, Policy Gradient, Actor-Critic).

use crate::error::{NumRs2Error, Result};
use crate::new_modules::rl::replay::Experience;
use crate::new_modules::rl::utils::RLAgent as RLAgentTrait;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::random::{Distribution, Rng, Uniform};
use std::collections::HashMap;

/// Re-export RLAgent trait
pub use crate::new_modules::rl::utils::RLAgent;

/// Q-Learning agent for tabular environments
///
/// Implements Q-Learning algorithm using table-based Q-values.
/// Suitable for discrete state/action spaces.
///
/// # Algorithm
///
/// ```text
/// Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
/// ```
///
/// where α is learning rate and γ is discount factor.
pub struct QLearningAgent {
    q_table: HashMap<Vec<u64>, Array1<f64>>,
    learning_rate: f64,
    gamma: f64,
    state_dim: usize,
    action_dim: usize,
}

impl QLearningAgent {
    /// Create new Q-Learning agent
    ///
    /// # Arguments
    /// * `state_dim` - State space dimensionality
    /// * `action_dim` - Number of discrete actions
    /// * `learning_rate` - Learning rate α
    /// * `gamma` - Discount factor γ
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        learning_rate: f64,
        gamma: f64,
    ) -> Result<Self> {
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(NumRs2Error::ValueError(
                "learning_rate must be in (0, 1]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(NumRs2Error::ValueError(
                "gamma must be in [0, 1]".to_string(),
            ));
        }

        Ok(Self {
            q_table: HashMap::new(),
            learning_rate,
            gamma,
            state_dim,
            action_dim,
        })
    }

    /// Discretize continuous state for table lookup
    fn discretize_state(&self, state: &Array1<f64>) -> Vec<u64> {
        state.iter().map(|&x| (x * 100.0) as u64).collect()
    }

    /// Get Q-values for state
    fn get_q_values(&self, state: &Array1<f64>) -> Array1<f64> {
        let discrete_state = self.discretize_state(state);
        self.q_table
            .get(&discrete_state)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(self.action_dim))
    }

    /// Update Q-table entry
    pub fn update(
        &mut self,
        state: &Array1<f64>,
        action: usize,
        reward: f64,
        next_state: &Array1<f64>,
        done: bool,
    ) -> Result<()> {
        let discrete_state = self.discretize_state(state);
        let mut q_values = self
            .q_table
            .get(&discrete_state)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(self.action_dim));

        let next_q_values = self.get_q_values(next_state);
        let max_next_q = if done {
            0.0
        } else {
            next_q_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        };

        let td_target = reward + self.gamma * max_next_q;
        let td_error = td_target - q_values[action];
        q_values[action] += self.learning_rate * td_error;

        self.q_table.insert(discrete_state, q_values);
        Ok(())
    }

    /// Get state space dimension
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Get learning rate
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Get discount factor
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
}

impl RLAgent for QLearningAgent {
    fn select_greedy_action(&self, state: &Array1<f64>) -> Result<usize> {
        let q_values = self.get_q_values(state);
        let (best_action, _) = q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| NumRs2Error::ValueError("No actions available".to_string()))?;
        Ok(best_action)
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }
}

/// SARSA agent (on-policy TD control)
///
/// Similar to Q-Learning but uses actual next action instead of max.
///
/// # Algorithm
///
/// ```text
/// Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
/// ```
///
/// where a' is the action actually taken (not necessarily optimal).
pub struct SARSAAgent {
    q_table: HashMap<Vec<u64>, Array1<f64>>,
    learning_rate: f64,
    gamma: f64,
    state_dim: usize,
    action_dim: usize,
}

impl SARSAAgent {
    /// Create new SARSA agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        learning_rate: f64,
        gamma: f64,
    ) -> Result<Self> {
        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(NumRs2Error::ValueError(
                "learning_rate must be in (0, 1]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(NumRs2Error::ValueError(
                "gamma must be in [0, 1]".to_string(),
            ));
        }

        Ok(Self {
            q_table: HashMap::new(),
            learning_rate,
            gamma,
            state_dim,
            action_dim,
        })
    }

    fn discretize_state(&self, state: &Array1<f64>) -> Vec<u64> {
        state.iter().map(|&x| (x * 100.0) as u64).collect()
    }

    fn get_q_values(&self, state: &Array1<f64>) -> Array1<f64> {
        let discrete_state = self.discretize_state(state);
        self.q_table
            .get(&discrete_state)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(self.action_dim))
    }

    /// Update using SARSA rule
    pub fn update(
        &mut self,
        state: &Array1<f64>,
        action: usize,
        reward: f64,
        next_state: &Array1<f64>,
        next_action: usize,
        done: bool,
    ) -> Result<()> {
        let discrete_state = self.discretize_state(state);
        let mut q_values = self
            .q_table
            .get(&discrete_state)
            .cloned()
            .unwrap_or_else(|| Array1::zeros(self.action_dim));

        let next_q_values = self.get_q_values(next_state);
        let next_q = if done {
            0.0
        } else {
            next_q_values[next_action]
        };

        let td_target = reward + self.gamma * next_q;
        let td_error = td_target - q_values[action];
        q_values[action] += self.learning_rate * td_error;

        self.q_table.insert(discrete_state, q_values);
        Ok(())
    }
}

impl RLAgent for SARSAAgent {
    fn select_greedy_action(&self, state: &Array1<f64>) -> Result<usize> {
        let q_values = self.get_q_values(state);
        let (best_action, _) = q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| NumRs2Error::ValueError("No actions available".to_string()))?;
        Ok(best_action)
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }
}

/// Deep Q-Network (DQN) agent
///
/// Uses neural network to approximate Q-values.
/// Includes experience replay and target network for stability.
///
/// # Algorithm
///
/// 1. Store transitions in replay buffer
/// 2. Sample mini-batch and compute loss:
///    ```text
///    L(θ) = E[(r + γ max_a' Q(s',a';θ⁻) - Q(s,a;θ))²]
///    ```
/// 3. Update Q-network using gradient descent
/// 4. Periodically update target network θ⁻ ← θ
pub struct DQNAgent {
    q_network: SimpleNetwork,
    target_network: SimpleNetwork,
    learning_rate: f64,
    gamma: f64,
    state_dim: usize,
    action_dim: usize,
}

impl DQNAgent {
    /// Create new DQN agent
    ///
    /// # Arguments
    /// * `state_dim` - State space dimensionality
    /// * `action_dim` - Number of discrete actions
    /// * `hidden_dims` - Hidden layer dimensions
    /// * `learning_rate` - Learning rate for network updates
    /// * `gamma` - Discount factor
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dims: Vec<usize>,
        learning_rate: f64,
        gamma: f64,
    ) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(NumRs2Error::ValueError(
                "learning_rate must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(NumRs2Error::ValueError(
                "gamma must be in [0, 1]".to_string(),
            ));
        }

        let q_network = SimpleNetwork::new(state_dim, action_dim, hidden_dims.clone())?;
        let target_network = SimpleNetwork::new(state_dim, action_dim, hidden_dims)?;

        Ok(Self {
            q_network,
            target_network,
            learning_rate,
            gamma,
            state_dim,
            action_dim,
        })
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action<R: Rng>(
        &self,
        state: &Array1<f64>,
        epsilon: f64,
        rng: &mut R,
    ) -> Result<usize> {
        let dist = Uniform::new(0.0, 1.0)
            .map_err(|e| NumRs2Error::ValueError(format!("Uniform distribution error: {}", e)))?;

        if dist.sample(rng) < epsilon {
            let action_dist = Uniform::new(0, self.action_dim).map_err(|e| {
                NumRs2Error::ValueError(format!("Uniform distribution error: {}", e))
            })?;
            Ok(action_dist.sample(rng))
        } else {
            self.select_greedy_action(state)
        }
    }

    /// Train on a batch of experiences
    pub fn train_batch(&mut self, batch: &[Experience]) -> Result<f64> {
        if batch.is_empty() {
            return Err(NumRs2Error::ValueError("Empty batch".to_string()));
        }

        let mut total_loss = 0.0;

        for exp in batch {
            // Compute Q(s, a)
            let q_values = self.q_network.forward(&exp.state)?;
            let q_value = q_values[exp.action];

            // Compute target: r + γ max_a' Q_target(s', a')
            let next_q_values = self.target_network.forward(&exp.next_state)?;
            let max_next_q = if exp.done {
                0.0
            } else {
                next_q_values
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            };

            let target = exp.reward + self.gamma * max_next_q;

            // Compute TD error and loss
            let td_error = target - q_value;
            let loss = td_error * td_error;
            total_loss += loss;

            // Update Q-network (simplified gradient update)
            self.q_network
                .update(&exp.state, exp.action, td_error, self.learning_rate)?;
        }

        Ok(total_loss / batch.len() as f64)
    }

    /// Update target network to match Q-network
    pub fn update_target_network(&mut self) -> Result<()> {
        self.target_network = self.q_network.clone();
        Ok(())
    }

    /// Soft update target network: θ⁻ ← τθ + (1-τ)θ⁻
    pub fn soft_update_target_network(&mut self, tau: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&tau) {
            return Err(NumRs2Error::ValueError("tau must be in [0, 1]".to_string()));
        }
        self.target_network.soft_update(&self.q_network, tau)?;
        Ok(())
    }
}

impl RLAgent for DQNAgent {
    fn select_greedy_action(&self, state: &Array1<f64>) -> Result<usize> {
        let q_values = self.q_network.forward(state)?;
        let (best_action, _) = q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| NumRs2Error::ValueError("No actions available".to_string()))?;
        Ok(best_action)
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Return Q-values for all actions given `state`.
    ///
    /// Runs a forward pass through the Q-network and converts the resulting
    /// `Array1<f64>` into a plain `Vec<f64>` so that exploration strategies
    /// (e.g. `BoltzmannExploration`) can compute action probabilities.
    fn action_values(&self, state: &Array1<f64>) -> Result<Vec<f64>> {
        let q_array = self.q_network.forward(state)?;
        Ok(q_array.to_vec())
    }
}

/// Policy Gradient (REINFORCE) agent
///
/// Directly learns policy π_θ(a|s) using policy gradient.
///
/// # Algorithm
///
/// ```text
/// ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * G_t]
/// ```
///
/// where G_t is the return (cumulative discounted reward).
pub struct PolicyGradientAgent {
    policy_network: SimpleNetwork,
    learning_rate: f64,
    gamma: f64,
    state_dim: usize,
    action_dim: usize,
}

impl PolicyGradientAgent {
    /// Create new policy gradient agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dims: Vec<usize>,
        learning_rate: f64,
        gamma: f64,
    ) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(NumRs2Error::ValueError(
                "learning_rate must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(NumRs2Error::ValueError(
                "gamma must be in [0, 1]".to_string(),
            ));
        }

        let policy_network = SimpleNetwork::new(state_dim, action_dim, hidden_dims)?;

        Ok(Self {
            policy_network,
            learning_rate,
            gamma,
            state_dim,
            action_dim,
        })
    }

    /// Select action from policy distribution
    pub fn select_action<R: Rng>(&self, state: &Array1<f64>, rng: &mut R) -> Result<usize> {
        let logits = self.policy_network.forward(state)?;
        let probs = softmax(&logits)?;

        // Sample from categorical distribution
        let dist = Uniform::new(0.0, 1.0)
            .map_err(|e| NumRs2Error::ValueError(format!("Uniform distribution error: {}", e)))?;
        let mut cumsum = 0.0;
        let sample = dist.sample(rng);

        for (action, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if sample <= cumsum {
                return Ok(action);
            }
        }

        Ok(self.action_dim - 1)
    }

    /// Train on episode trajectory
    pub fn train_episode(&mut self, trajectory: &[(Array1<f64>, usize, f64)]) -> Result<f64> {
        if trajectory.is_empty() {
            return Err(NumRs2Error::ValueError("Empty trajectory".to_string()));
        }

        // Compute returns
        let mut returns = vec![0.0; trajectory.len()];
        let mut g = 0.0;
        for (i, (_, _, reward)) in trajectory.iter().enumerate().rev() {
            g = reward + self.gamma * g;
            returns[i] = g;
        }

        // Normalize returns for stability
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64)
            .sqrt()
            + 1e-8;
        let normalized_returns: Vec<f64> = returns.iter().map(|r| (r - mean) / std).collect();

        // Update policy
        let mut total_loss = 0.0;
        for ((state, action, _), &return_val) in trajectory.iter().zip(normalized_returns.iter()) {
            let loss =
                self.policy_network
                    .update(state, *action, return_val, self.learning_rate)?;
            total_loss += loss;
        }

        Ok(total_loss / trajectory.len() as f64)
    }
}

impl RLAgent for PolicyGradientAgent {
    fn select_greedy_action(&self, state: &Array1<f64>) -> Result<usize> {
        let logits = self.policy_network.forward(state)?;
        let (best_action, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| NumRs2Error::ValueError("No actions available".to_string()))?;
        Ok(best_action)
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }
}

/// Actor-Critic agent
///
/// Combines policy gradient (actor) with value function learning (critic).
///
/// # Algorithm
///
/// Actor update:
/// ```text
/// ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]
/// ```
///
/// Critic update:
/// ```text
/// L(φ) = E[(V_φ(s) - (r + γV_φ(s')))²]
/// ```
///
/// where A(s,a) = r + γV(s') - V(s) is the advantage.
pub struct ActorCriticAgent {
    actor_network: SimpleNetwork,
    critic_network: SimpleNetwork,
    actor_lr: f64,
    critic_lr: f64,
    gamma: f64,
    state_dim: usize,
    action_dim: usize,
}

impl ActorCriticAgent {
    /// Create new actor-critic agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        actor_hidden_dims: Vec<usize>,
        critic_hidden_dims: Vec<usize>,
        actor_lr: f64,
        critic_lr: f64,
        gamma: f64,
    ) -> Result<Self> {
        if actor_lr <= 0.0 {
            return Err(NumRs2Error::ValueError(
                "actor_lr must be positive".to_string(),
            ));
        }
        if critic_lr <= 0.0 {
            return Err(NumRs2Error::ValueError(
                "critic_lr must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&gamma) {
            return Err(NumRs2Error::ValueError(
                "gamma must be in [0, 1]".to_string(),
            ));
        }

        let actor_network = SimpleNetwork::new(state_dim, action_dim, actor_hidden_dims)?;
        let critic_network = SimpleNetwork::new(state_dim, 1, critic_hidden_dims)?;

        Ok(Self {
            actor_network,
            critic_network,
            actor_lr,
            critic_lr,
            gamma,
            state_dim,
            action_dim,
        })
    }

    /// Select action from policy
    pub fn select_action<R: Rng>(&self, state: &Array1<f64>, rng: &mut R) -> Result<usize> {
        let logits = self.actor_network.forward(state)?;
        let probs = softmax(&logits)?;

        let dist = Uniform::new(0.0, 1.0)
            .map_err(|e| NumRs2Error::ValueError(format!("Uniform distribution error: {}", e)))?;
        let mut cumsum = 0.0;
        let sample = dist.sample(rng);

        for (action, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if sample <= cumsum {
                return Ok(action);
            }
        }

        Ok(self.action_dim - 1)
    }

    /// Train on a single step
    pub fn train_step(
        &mut self,
        state: &Array1<f64>,
        action: usize,
        reward: f64,
        next_state: &Array1<f64>,
        done: bool,
    ) -> Result<(f64, f64)> {
        // Compute value estimates
        let value = self.critic_network.forward(state)?[0];
        let next_value = if done {
            0.0
        } else {
            self.critic_network.forward(next_state)?[0]
        };

        // Compute advantage
        let advantage = reward + self.gamma * next_value - value;

        // Update critic
        let critic_loss = self
            .critic_network
            .update(state, 0, advantage, self.critic_lr)?;

        // Update actor
        let actor_loss = self
            .actor_network
            .update(state, action, advantage, self.actor_lr)?;

        Ok((actor_loss, critic_loss))
    }
}

impl RLAgent for ActorCriticAgent {
    fn select_greedy_action(&self, state: &Array1<f64>) -> Result<usize> {
        let logits = self.actor_network.forward(state)?;
        let (best_action, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| NumRs2Error::ValueError("No actions available".to_string()))?;
        Ok(best_action)
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }
}

/// Simple feedforward neural network for RL agents
///
/// Basic implementation for demonstration. In production, use full NN module.
#[derive(Clone)]
struct SimpleNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

impl SimpleNetwork {
    fn new(input_dim: usize, output_dim: usize, hidden_dims: Vec<usize>) -> Result<Self> {
        use scirs2_core::random::thread_rng;

        let mut layer_dims = vec![input_dim];
        layer_dims.extend(hidden_dims);
        layer_dims.push(output_dim);

        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut rng = thread_rng();

        for i in 0..layer_dims.len() - 1 {
            let dist = Uniform::new(-0.01, 0.01).map_err(|e| {
                NumRs2Error::ValueError(format!("Uniform distribution error: {}", e))
            })?;

            let w = Array2::from_shape_fn((layer_dims[i], layer_dims[i + 1]), |_| {
                dist.sample(&mut rng)
            });
            let b = Array1::zeros(layer_dims[i + 1]);
            weights.push(w);
            biases.push(b);
        }

        Ok(Self { weights, biases })
    }

    fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let mut activation = input.clone();

        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            // Matrix-vector multiplication
            let mut output = Array1::zeros(w.ncols());
            for (row_idx, row) in w.axis_iter(Axis(0)).enumerate() {
                for (col_idx, &val) in row.iter().enumerate() {
                    output[col_idx] += activation[row_idx] * val;
                }
            }

            // Add bias
            output = &output + b;

            // Apply ReLU activation (except last layer)
            if i < self.weights.len() - 1 {
                activation = output.mapv(|x: f64| x.max(0.0));
            } else {
                activation = output;
            }
        }

        Ok(activation)
    }

    fn update(
        &mut self,
        state: &Array1<f64>,
        action: usize,
        gradient_signal: f64,
        learning_rate: f64,
    ) -> Result<f64> {
        // Simplified gradient update (placeholder)
        // In production, use full backpropagation
        let output = self.forward(state)?;

        if action >= output.len() {
            return Err(NumRs2Error::ValueError(format!(
                "Action {} out of bounds for output size {}",
                action,
                output.len()
            )));
        }

        // Simple update: adjust weights based on gradient signal
        for (w, b) in self.weights.iter_mut().zip(self.biases.iter_mut()) {
            for val in w.iter_mut() {
                *val += learning_rate * gradient_signal * 0.01;
            }
            for val in b.iter_mut() {
                *val += learning_rate * gradient_signal * 0.01;
            }
        }

        Ok(gradient_signal.abs())
    }

    fn soft_update(&mut self, other: &SimpleNetwork, tau: f64) -> Result<()> {
        for (w_target, w_source) in self.weights.iter_mut().zip(other.weights.iter()) {
            *w_target = &(w_target.clone() * (1.0 - tau)) + &(w_source * tau);
        }
        for (b_target, b_source) in self.biases.iter_mut().zip(other.biases.iter()) {
            *b_target = &(b_target.clone() * (1.0 - tau)) + &(b_source * tau);
        }
        Ok(())
    }
}

/// Softmax function
fn softmax(x: &Array1<f64>) -> Result<Array1<f64>> {
    let max_x = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x: Array1<f64> = x.mapv(|v| (v - max_x).exp());
    let sum_exp_x: f64 = exp_x.sum();

    if sum_exp_x == 0.0 || !sum_exp_x.is_finite() {
        return Err(NumRs2Error::NumericalError(
            "Softmax computation failed".to_string(),
        ));
    }

    Ok(exp_x / sum_exp_x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_qlearning_creation() -> Result<()> {
        let agent = QLearningAgent::new(4, 2, 0.1, 0.99)?;
        assert_eq!(agent.state_dim(), 4);
        assert_eq!(agent.action_dim(), 2);
        assert_eq!(agent.learning_rate(), 0.1);
        assert_eq!(agent.gamma(), 0.99);
        Ok(())
    }

    #[test]
    fn test_qlearning_invalid_params() {
        assert!(QLearningAgent::new(4, 2, 0.0, 0.99).is_err());
        assert!(QLearningAgent::new(4, 2, 1.5, 0.99).is_err());
        assert!(QLearningAgent::new(4, 2, 0.1, -0.1).is_err());
        assert!(QLearningAgent::new(4, 2, 0.1, 1.5).is_err());
    }

    #[test]
    fn test_qlearning_select_action() -> Result<()> {
        let agent = QLearningAgent::new(2, 3, 0.1, 0.99)?;
        let state = Array1::from_vec(vec![0.5, 0.5]);
        let action = agent.select_greedy_action(&state)?;
        assert!(action < 3);
        Ok(())
    }

    #[test]
    fn test_qlearning_update() -> Result<()> {
        let mut agent = QLearningAgent::new(2, 3, 0.1, 0.99)?;
        let state = Array1::from_vec(vec![0.5, 0.5]);
        let next_state = Array1::from_vec(vec![0.6, 0.6]);

        agent.update(&state, 0, 1.0, &next_state, false)?;
        Ok(())
    }

    #[test]
    fn test_sarsa_creation() -> Result<()> {
        let agent = SARSAAgent::new(4, 2, 0.1, 0.99)?;
        assert_eq!(agent.action_dim(), 2);
        Ok(())
    }

    #[test]
    fn test_sarsa_update() -> Result<()> {
        let mut agent = SARSAAgent::new(2, 3, 0.1, 0.99)?;
        let state = Array1::from_vec(vec![0.5, 0.5]);
        let next_state = Array1::from_vec(vec![0.6, 0.6]);

        agent.update(&state, 0, 1.0, &next_state, 1, false)?;
        Ok(())
    }

    #[test]
    fn test_dqn_creation() -> Result<()> {
        let agent = DQNAgent::new(4, 2, vec![16], 0.001, 0.99)?;
        assert_eq!(agent.action_dim(), 2);
        Ok(())
    }

    #[test]
    fn test_dqn_select_action() -> Result<()> {
        let agent = DQNAgent::new(4, 2, vec![16], 0.001, 0.99)?;
        let mut rng = thread_rng();
        let state = Array1::from_vec(vec![0.5, 0.5, 0.0, 0.0]);

        let action = agent.select_action(&state, 0.1, &mut rng)?;
        assert!(action < 2);
        Ok(())
    }

    #[test]
    fn test_dqn_train_batch() -> Result<()> {
        let mut agent = DQNAgent::new(2, 2, vec![8], 0.001, 0.99)?;

        let batch = vec![
            Experience {
                state: Array1::from_vec(vec![0.5, 0.5]),
                action: 0,
                reward: 1.0,
                next_state: Array1::from_vec(vec![0.6, 0.6]),
                done: false,
            },
            Experience {
                state: Array1::from_vec(vec![0.6, 0.6]),
                action: 1,
                reward: 0.5,
                next_state: Array1::from_vec(vec![0.7, 0.7]),
                done: true,
            },
        ];

        let loss = agent.train_batch(&batch)?;
        assert!(loss >= 0.0);
        Ok(())
    }

    #[test]
    fn test_dqn_update_target() -> Result<()> {
        let mut agent = DQNAgent::new(2, 2, vec![8], 0.001, 0.99)?;
        agent.update_target_network()?;
        Ok(())
    }

    #[test]
    fn test_dqn_soft_update() -> Result<()> {
        let mut agent = DQNAgent::new(2, 2, vec![8], 0.001, 0.99)?;
        agent.soft_update_target_network(0.01)?;
        Ok(())
    }

    #[test]
    fn test_policy_gradient_creation() -> Result<()> {
        let agent = PolicyGradientAgent::new(4, 2, vec![16], 0.001, 0.99)?;
        assert_eq!(agent.action_dim(), 2);
        Ok(())
    }

    #[test]
    fn test_policy_gradient_select_action() -> Result<()> {
        let agent = PolicyGradientAgent::new(4, 2, vec![16], 0.001, 0.99)?;
        let mut rng = thread_rng();
        let state = Array1::from_vec(vec![0.5, 0.5, 0.0, 0.0]);

        let action = agent.select_action(&state, &mut rng)?;
        assert!(action < 2);
        Ok(())
    }

    #[test]
    fn test_policy_gradient_train_episode() -> Result<()> {
        let mut agent = PolicyGradientAgent::new(2, 2, vec![8], 0.001, 0.99)?;

        let trajectory = vec![
            (Array1::from_vec(vec![0.5, 0.5]), 0, 1.0),
            (Array1::from_vec(vec![0.6, 0.6]), 1, 0.5),
            (Array1::from_vec(vec![0.7, 0.7]), 0, 0.2),
        ];

        let loss = agent.train_episode(&trajectory)?;
        assert!(loss >= 0.0);
        Ok(())
    }

    #[test]
    fn test_actor_critic_creation() -> Result<()> {
        let agent = ActorCriticAgent::new(4, 2, vec![16], vec![16], 0.001, 0.001, 0.99)?;
        assert_eq!(agent.action_dim(), 2);
        Ok(())
    }

    #[test]
    fn test_actor_critic_select_action() -> Result<()> {
        let agent = ActorCriticAgent::new(4, 2, vec![16], vec![16], 0.001, 0.001, 0.99)?;
        let mut rng = thread_rng();
        let state = Array1::from_vec(vec![0.5, 0.5, 0.0, 0.0]);

        let action = agent.select_action(&state, &mut rng)?;
        assert!(action < 2);
        Ok(())
    }

    #[test]
    fn test_actor_critic_train_step() -> Result<()> {
        let mut agent = ActorCriticAgent::new(2, 2, vec![8], vec![8], 0.001, 0.001, 0.99)?;

        let state = Array1::from_vec(vec![0.5, 0.5]);
        let next_state = Array1::from_vec(vec![0.6, 0.6]);

        let (actor_loss, critic_loss) = agent.train_step(&state, 0, 1.0, &next_state, false)?;
        assert!(actor_loss >= 0.0);
        assert!(critic_loss >= 0.0);
        Ok(())
    }

    #[test]
    fn test_simple_network_creation() -> Result<()> {
        let network = SimpleNetwork::new(4, 2, vec![16, 16])?;
        assert_eq!(network.weights.len(), 3); // input->hidden1, hidden1->hidden2, hidden2->output
        Ok(())
    }

    #[test]
    fn test_simple_network_forward() -> Result<()> {
        let network = SimpleNetwork::new(4, 2, vec![8])?;
        let input = Array1::from_vec(vec![0.5, 0.5, 0.0, 0.0]);
        let output = network.forward(&input)?;
        assert_eq!(output.len(), 2);
        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<()> {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&x)?;

        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
        Ok(())
    }
}
