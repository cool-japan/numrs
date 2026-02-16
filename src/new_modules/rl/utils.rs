//! Utilities for Reinforcement Learning
//!
//! This module provides exploration strategies, reward normalization,
//! and episode tracking utilities for RL agents.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{Distribution, Rng, Uniform};

/// Trait for exploration strategies
pub trait ExplorationStrategy {
    /// Select action using exploration strategy
    fn select_action<A: RLAgent, R: Rng>(
        &self,
        agent: &A,
        state: &Array1<f64>,
        rng: &mut R,
    ) -> Result<usize>;

    /// Decay exploration parameter (if applicable)
    fn decay(&mut self);

    /// Get current exploration parameter
    fn exploration_param(&self) -> f64;
}

/// Placeholder trait for agents (will be defined in agents.rs)
pub trait RLAgent {
    /// Select greedy action
    fn select_greedy_action(&self, state: &Array1<f64>) -> Result<usize>;

    /// Get number of actions
    fn action_dim(&self) -> usize;
}

/// Epsilon-greedy exploration strategy
///
/// With probability epsilon, selects a random action.
/// With probability 1-epsilon, selects the greedy action.
///
/// Epsilon decays over time: epsilon = max(epsilon_min, epsilon * decay_rate)
pub struct EpsilonGreedy {
    epsilon: f64,
    epsilon_min: f64,
    decay_rate: f64,
}

impl EpsilonGreedy {
    /// Create new epsilon-greedy strategy
    ///
    /// # Arguments
    /// * `epsilon` - Initial exploration probability
    /// * `epsilon_min` - Minimum exploration probability
    /// * `decay_rate` - Multiplicative decay rate per step
    pub fn new(epsilon: f64, epsilon_min: f64, decay_rate: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&epsilon) {
            return Err(NumRs2Error::ValueError(
                "epsilon must be in [0, 1]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&epsilon_min) {
            return Err(NumRs2Error::ValueError(
                "epsilon_min must be in [0, 1]".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&decay_rate) {
            return Err(NumRs2Error::ValueError(
                "decay_rate must be in [0, 1]".to_string(),
            ));
        }

        Ok(Self {
            epsilon,
            epsilon_min,
            decay_rate,
        })
    }

    /// Get current epsilon value
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl ExplorationStrategy for EpsilonGreedy {
    fn select_action<A: RLAgent, R: Rng>(
        &self,
        agent: &A,
        state: &Array1<f64>,
        rng: &mut R,
    ) -> Result<usize> {
        let dist = Uniform::new(0.0, 1.0)
            .map_err(|e| NumRs2Error::ValueError(format!("Uniform distribution error: {}", e)))?;

        if dist.sample(rng) < self.epsilon {
            // Explore: random action
            let action_dist = Uniform::new(0, agent.action_dim()).map_err(|e| {
                NumRs2Error::ValueError(format!("Uniform distribution error: {}", e))
            })?;
            Ok(action_dist.sample(rng))
        } else {
            // Exploit: greedy action
            agent.select_greedy_action(state)
        }
    }

    fn decay(&mut self) {
        self.epsilon = (self.epsilon * self.decay_rate).max(self.epsilon_min);
    }

    fn exploration_param(&self) -> f64 {
        self.epsilon
    }
}

/// Boltzmann (softmax) exploration strategy
///
/// Selects actions probabilistically based on their values:
/// P(a|s) ∝ exp(Q(s,a) / temperature)
///
/// Higher temperature = more exploration
/// Lower temperature = more exploitation
pub struct BoltzmannExploration {
    temperature: f64,
    temperature_min: f64,
    decay_rate: f64,
}

impl BoltzmannExploration {
    /// Create new Boltzmann exploration strategy
    ///
    /// # Arguments
    /// * `temperature` - Initial temperature parameter
    /// * `temperature_min` - Minimum temperature
    /// * `decay_rate` - Multiplicative decay rate per step
    pub fn new(temperature: f64, temperature_min: f64, decay_rate: f64) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(NumRs2Error::ValueError(
                "temperature must be positive".to_string(),
            ));
        }
        if temperature_min <= 0.0 {
            return Err(NumRs2Error::ValueError(
                "temperature_min must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&decay_rate) {
            return Err(NumRs2Error::ValueError(
                "decay_rate must be in [0, 1]".to_string(),
            ));
        }

        Ok(Self {
            temperature,
            temperature_min,
            decay_rate,
        })
    }

    /// Get current temperature value
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Compute softmax probabilities for action values
    fn softmax(&self, values: &[f64]) -> Result<Vec<f64>> {
        if values.is_empty() {
            return Err(NumRs2Error::ValueError(
                "Cannot compute softmax of empty array".to_string(),
            ));
        }

        // Subtract max for numerical stability
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f64> = values
            .iter()
            .map(|&v| ((v - max_val) / self.temperature).exp())
            .collect();

        let sum: f64 = exp_values.iter().sum();
        if sum == 0.0 || !sum.is_finite() {
            return Err(NumRs2Error::NumericalError(
                "Softmax computation resulted in invalid sum".to_string(),
            ));
        }

        Ok(exp_values.iter().map(|&v| v / sum).collect())
    }
}

impl ExplorationStrategy for BoltzmannExploration {
    fn select_action<A: RLAgent, R: Rng>(
        &self,
        agent: &A,
        state: &Array1<f64>,
        _rng: &mut R,
    ) -> Result<usize> {
        // This is a placeholder - actual implementation would need agent to return Q-values
        // For now, fall back to greedy selection
        agent.select_greedy_action(state)
    }

    fn decay(&mut self) {
        self.temperature = (self.temperature * self.decay_rate).max(self.temperature_min);
    }

    fn exploration_param(&self) -> f64 {
        self.temperature
    }
}

/// Reward normalizer using running statistics
///
/// Normalizes rewards to have zero mean and unit variance using
/// Welford's online algorithm for numerical stability.
pub struct RewardNormalizer {
    mean: f64,
    var: f64,
    count: usize,
    epsilon: f64,
}

impl RewardNormalizer {
    /// Create new reward normalizer
    ///
    /// # Arguments
    /// * `epsilon` - Small constant for numerical stability (default: 1e-8)
    pub fn new(epsilon: f64) -> Self {
        Self {
            mean: 0.0,
            var: 1.0,
            count: 0,
            epsilon,
        }
    }

    /// Update statistics with new reward
    pub fn update(&mut self, reward: f64) {
        self.count += 1;
        let delta = reward - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = reward - self.mean;
        self.var += delta * delta2;
    }

    /// Normalize reward
    pub fn normalize(&self, reward: f64) -> f64 {
        if self.count < 2 {
            return reward;
        }
        let std = (self.var / (self.count - 1) as f64).sqrt() + self.epsilon;
        (reward - self.mean) / std
    }

    /// Get current mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current standard deviation
    pub fn std(&self) -> f64 {
        if self.count < 2 {
            return 1.0;
        }
        (self.var / (self.count - 1) as f64).sqrt()
    }

    /// Get sample count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.var = 1.0;
        self.count = 0;
    }
}

impl Default for RewardNormalizer {
    fn default() -> Self {
        Self::new(1e-8)
    }
}

/// Episode statistics tracker
///
/// Tracks cumulative rewards, episode lengths, and other statistics
/// across training episodes.
#[derive(Debug, Clone)]
pub struct EpisodeTracker {
    episode_rewards: Vec<f64>,
    episode_lengths: Vec<usize>,
    current_episode_reward: f64,
    current_episode_length: usize,
    window_size: usize,
}

impl EpisodeTracker {
    /// Create new episode tracker
    ///
    /// # Arguments
    /// * `window_size` - Number of recent episodes to keep for moving average
    pub fn new(window_size: usize) -> Self {
        Self {
            episode_rewards: Vec::new(),
            episode_lengths: Vec::new(),
            current_episode_reward: 0.0,
            current_episode_length: 0,
            window_size,
        }
    }

    /// Record step in current episode
    pub fn step(&mut self, reward: f64) {
        self.current_episode_reward += reward;
        self.current_episode_length += 1;
    }

    /// Finish current episode and record statistics
    pub fn finish_episode(&mut self) {
        self.episode_rewards.push(self.current_episode_reward);
        self.episode_lengths.push(self.current_episode_length);

        // Keep only recent episodes within window
        if self.episode_rewards.len() > self.window_size {
            self.episode_rewards.remove(0);
            self.episode_lengths.remove(0);
        }

        self.current_episode_reward = 0.0;
        self.current_episode_length = 0;
    }

    /// Get total number of episodes
    pub fn num_episodes(&self) -> usize {
        self.episode_rewards.len()
    }

    /// Get mean reward over recent episodes
    pub fn mean_reward(&self) -> Option<f64> {
        if self.episode_rewards.is_empty() {
            return None;
        }
        let sum: f64 = self.episode_rewards.iter().sum();
        Some(sum / self.episode_rewards.len() as f64)
    }

    /// Get mean episode length over recent episodes
    pub fn mean_length(&self) -> Option<f64> {
        if self.episode_lengths.is_empty() {
            return None;
        }
        let sum: usize = self.episode_lengths.iter().sum();
        Some(sum as f64 / self.episode_lengths.len() as f64)
    }

    /// Get last episode reward
    pub fn last_reward(&self) -> Option<f64> {
        self.episode_rewards.last().copied()
    }

    /// Get last episode length
    pub fn last_length(&self) -> Option<usize> {
        self.episode_lengths.last().copied()
    }

    /// Get all episode rewards
    pub fn episode_rewards(&self) -> &[f64] {
        &self.episode_rewards
    }

    /// Get all episode lengths
    pub fn episode_lengths(&self) -> &[usize] {
        &self.episode_lengths
    }

    /// Reset tracker
    pub fn reset(&mut self) {
        self.episode_rewards.clear();
        self.episode_lengths.clear();
        self.current_episode_reward = 0.0;
        self.current_episode_length = 0;
    }
}

impl Default for EpisodeTracker {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::thread_rng;

    struct DummyAgent {
        action_dim: usize,
    }

    impl RLAgent for DummyAgent {
        fn select_greedy_action(&self, _state: &Array1<f64>) -> Result<usize> {
            Ok(0)
        }

        fn action_dim(&self) -> usize {
            self.action_dim
        }
    }

    #[test]
    fn test_epsilon_greedy_creation() -> Result<()> {
        let strategy = EpsilonGreedy::new(1.0, 0.01, 0.995)?;
        assert_eq!(strategy.epsilon(), 1.0);
        Ok(())
    }

    #[test]
    fn test_epsilon_greedy_invalid_params() {
        assert!(EpsilonGreedy::new(1.5, 0.01, 0.995).is_err());
        assert!(EpsilonGreedy::new(-0.1, 0.01, 0.995).is_err());
        assert!(EpsilonGreedy::new(1.0, 1.5, 0.995).is_err());
        assert!(EpsilonGreedy::new(1.0, 0.01, 1.5).is_err());
    }

    #[test]
    fn test_epsilon_greedy_decay() -> Result<()> {
        let mut strategy = EpsilonGreedy::new(1.0, 0.01, 0.9)?;
        strategy.decay();
        assert!((strategy.epsilon() - 0.9).abs() < 1e-6);

        for _ in 0..100 {
            strategy.decay();
        }
        assert!(strategy.epsilon() >= 0.01);
        Ok(())
    }

    #[test]
    fn test_epsilon_greedy_action_selection() -> Result<()> {
        let agent = DummyAgent { action_dim: 4 };
        let mut rng = thread_rng();
        let state = Array1::zeros(2);

        // Test with epsilon = 0 (always greedy)
        let strategy = EpsilonGreedy::new(0.0, 0.0, 1.0)?;
        let action = strategy.select_action(&agent, &state, &mut rng)?;
        assert_eq!(action, 0); // Should always select greedy action

        // Test with epsilon = 1 (always random)
        let strategy = EpsilonGreedy::new(1.0, 1.0, 1.0)?;
        let action = strategy.select_action(&agent, &state, &mut rng)?;
        assert!(action < 4); // Should be valid action

        Ok(())
    }

    #[test]
    fn test_boltzmann_creation() -> Result<()> {
        let strategy = BoltzmannExploration::new(1.0, 0.1, 0.99)?;
        assert_eq!(strategy.temperature(), 1.0);
        Ok(())
    }

    #[test]
    fn test_boltzmann_invalid_params() {
        assert!(BoltzmannExploration::new(0.0, 0.1, 0.99).is_err());
        assert!(BoltzmannExploration::new(-1.0, 0.1, 0.99).is_err());
        assert!(BoltzmannExploration::new(1.0, 0.0, 0.99).is_err());
        assert!(BoltzmannExploration::new(1.0, 0.1, 1.5).is_err());
    }

    #[test]
    fn test_boltzmann_decay() -> Result<()> {
        let mut strategy = BoltzmannExploration::new(10.0, 0.1, 0.9)?;
        strategy.decay();
        assert!((strategy.temperature() - 9.0).abs() < 1e-6);

        for _ in 0..200 {
            strategy.decay();
        }
        assert!(strategy.temperature() >= 0.1);
        Ok(())
    }

    #[test]
    fn test_boltzmann_softmax() -> Result<()> {
        let strategy = BoltzmannExploration::new(1.0, 0.1, 0.99)?;
        let values = vec![1.0, 2.0, 3.0];
        let probs = strategy.softmax(&values)?;

        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
        Ok(())
    }

    #[test]
    fn test_boltzmann_softmax_empty() -> Result<()> {
        let strategy = BoltzmannExploration::new(1.0, 0.1, 0.99)?;
        let values: Vec<f64> = vec![];
        let result = strategy.softmax(&values);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_reward_normalizer_creation() {
        let normalizer = RewardNormalizer::new(1e-8);
        assert_eq!(normalizer.mean(), 0.0);
        assert_eq!(normalizer.count(), 0);
    }

    #[test]
    fn test_reward_normalizer_update() {
        let mut normalizer = RewardNormalizer::new(1e-8);
        normalizer.update(1.0);
        normalizer.update(2.0);
        normalizer.update(3.0);

        assert_eq!(normalizer.count(), 3);
        assert!((normalizer.mean() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_reward_normalizer_normalize() {
        let mut normalizer = RewardNormalizer::new(1e-8);

        // Add samples with mean=5, std=2
        for i in 1..=5 {
            normalizer.update(i as f64);
        }

        let normalized = normalizer.normalize(3.0);
        assert!((normalized - 0.0).abs() < 0.5); // Should be close to 0 (mean)
    }

    #[test]
    fn test_reward_normalizer_reset() {
        let mut normalizer = RewardNormalizer::new(1e-8);
        normalizer.update(1.0);
        normalizer.update(2.0);
        normalizer.reset();

        assert_eq!(normalizer.count(), 0);
        assert_eq!(normalizer.mean(), 0.0);
    }

    #[test]
    fn test_episode_tracker_creation() {
        let tracker = EpisodeTracker::new(100);
        assert_eq!(tracker.num_episodes(), 0);
        assert!(tracker.mean_reward().is_none());
    }

    #[test]
    fn test_episode_tracker_step() {
        let mut tracker = EpisodeTracker::new(100);
        tracker.step(1.0);
        tracker.step(2.0);
        tracker.step(3.0);
        tracker.finish_episode();

        assert_eq!(tracker.num_episodes(), 1);
        assert_eq!(tracker.last_reward(), Some(6.0));
        assert_eq!(tracker.last_length(), Some(3));
    }

    #[test]
    fn test_episode_tracker_multiple_episodes() {
        let mut tracker = EpisodeTracker::new(100);

        for _ in 0..3 {
            for i in 1..=10 {
                tracker.step(i as f64);
            }
            tracker.finish_episode();
        }

        assert_eq!(tracker.num_episodes(), 3);
        assert_eq!(tracker.mean_reward(), Some(55.0));
        assert_eq!(tracker.mean_length(), Some(10.0));
    }

    #[test]
    fn test_episode_tracker_window() {
        let mut tracker = EpisodeTracker::new(2);

        for episode in 1..=5 {
            tracker.step(episode as f64);
            tracker.finish_episode();
        }

        assert_eq!(tracker.num_episodes(), 2); // Only keep last 2
        assert_eq!(tracker.last_reward(), Some(5.0));
        assert_eq!(tracker.mean_reward(), Some(4.5)); // (4 + 5) / 2
    }

    #[test]
    fn test_episode_tracker_reset() {
        let mut tracker = EpisodeTracker::new(100);
        tracker.step(1.0);
        tracker.finish_episode();
        tracker.reset();

        assert_eq!(tracker.num_episodes(), 0);
        assert!(tracker.mean_reward().is_none());
    }

    #[test]
    fn test_episode_tracker_accessors() {
        let mut tracker = EpisodeTracker::new(100);

        for i in 1..=3 {
            tracker.step(i as f64);
            tracker.step(i as f64);
            tracker.finish_episode();
        }

        let rewards = tracker.episode_rewards();
        assert_eq!(rewards, &[2.0, 4.0, 6.0]);

        let lengths = tracker.episode_lengths();
        assert_eq!(lengths, &[2, 2, 2]);
    }
}
