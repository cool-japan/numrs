//! Experience Replay Buffers for Deep RL
//!
//! This module provides experience replay buffers for deep reinforcement learning,
//! including standard replay and prioritized experience replay.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{Distribution, Rng, Uniform};
use std::collections::VecDeque;

/// Experience tuple for replay buffer
#[derive(Debug, Clone)]
pub struct Experience {
    /// Current state
    pub state: Array1<f64>,
    /// Action taken
    pub action: usize,
    /// Reward received
    pub reward: f64,
    /// Next state
    pub next_state: Array1<f64>,
    /// Whether episode is done
    pub done: bool,
}

/// Standard experience replay buffer
///
/// Stores experiences in a circular buffer and samples uniformly at random.
/// Used in DQN and other off-policy algorithms.
///
/// # Mathematical Background
///
/// Experience replay breaks correlation between consecutive samples by storing
/// transitions (s, a, r, s', done) and sampling randomly from the buffer.
///
/// This provides several benefits:
/// - Breaks temporal correlation in data
/// - Improves data efficiency through reuse
/// - Smooths learning distribution
pub struct ExperienceReplay {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ExperienceReplay {
    /// Create new experience replay buffer
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of experiences to store
    pub fn new(capacity: usize) -> Self {
        if capacity == 0 {
            panic!("Capacity must be positive");
        }

        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add experience to buffer
    ///
    /// If buffer is full, oldest experience is removed.
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Sample batch of experiences uniformly at random
    ///
    /// # Arguments
    /// * `batch_size` - Number of experiences to sample
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Vector of sampled experiences
    pub fn sample<R: Rng>(&self, batch_size: usize, rng: &mut R) -> Result<Vec<Experience>> {
        if batch_size > self.buffer.len() {
            return Err(NumRs2Error::ValueError(format!(
                "Requested batch size {} exceeds buffer size {}",
                batch_size,
                self.buffer.len()
            )));
        }

        let dist = Uniform::new(0, self.buffer.len())
            .map_err(|e| NumRs2Error::ValueError(format!("Uniform distribution error: {}", e)))?;

        let mut samples = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = dist.sample(rng);
            samples.push(self.buffer[idx].clone());
        }

        Ok(samples)
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Get reference to all experiences
    pub fn experiences(&self) -> &VecDeque<Experience> {
        &self.buffer
    }
}

/// Prioritized experience replay buffer
///
/// Samples experiences based on their TD-error priority, allowing the agent
/// to learn more from surprising transitions.
///
/// # Mathematical Background
///
/// Experiences are sampled with probability:
///
/// ```text
/// P(i) = p_i^α / Σ_k p_k^α
/// ```
///
/// where p_i is the priority of experience i and α controls the amount of
/// prioritization (α=0 gives uniform sampling).
///
/// To correct for non-uniform sampling, importance-sampling weights are used:
///
/// ```text
/// w_i = (N * P(i))^(-β) / max_j w_j
/// ```
///
/// where β is annealed from β_start to 1 over training.
///
/// # References
///
/// Schaul et al. (2015). "Prioritized Experience Replay"
/// arXiv:1511.05952
pub struct PrioritizedExperienceReplay {
    buffer: VecDeque<Experience>,
    priorities: VecDeque<f64>,
    capacity: usize,
    alpha: f64,
    beta: f64,
    beta_increment: f64,
    epsilon: f64,
    max_priority: f64,
}

impl PrioritizedExperienceReplay {
    /// Create new prioritized experience replay buffer
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of experiences to store
    /// * `alpha` - Priority exponent (α). Higher values prioritize more. Typical: 0.6
    /// * `beta` - Initial importance-sampling exponent (β). Typical: 0.4
    /// * `beta_increment` - Amount to increment β per sample. Typical: 1e-6
    /// * `epsilon` - Small constant to ensure non-zero priorities. Typical: 1e-6
    pub fn new(
        capacity: usize,
        alpha: f64,
        beta: f64,
        beta_increment: f64,
        epsilon: f64,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(NumRs2Error::ValueError(
                "Capacity must be positive".to_string(),
            ));
        }
        if alpha < 0.0 {
            return Err(NumRs2Error::ValueError(
                "alpha must be non-negative".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&beta) {
            return Err(NumRs2Error::ValueError(
                "beta must be in [0, 1]".to_string(),
            ));
        }
        if epsilon <= 0.0 {
            return Err(NumRs2Error::ValueError(
                "epsilon must be positive".to_string(),
            ));
        }

        Ok(Self {
            buffer: VecDeque::with_capacity(capacity),
            priorities: VecDeque::with_capacity(capacity),
            capacity,
            alpha,
            beta,
            beta_increment,
            epsilon,
            max_priority: 1.0,
        })
    }

    /// Add experience to buffer with initial max priority
    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
            self.priorities.pop_front();
        }

        self.buffer.push_back(experience);
        self.priorities.push_back(self.max_priority);
    }

    /// Sample batch of experiences based on priorities
    ///
    /// # Returns
    /// Tuple of (experiences, importance_sampling_weights, indices)
    pub fn sample<R: Rng>(
        &mut self,
        batch_size: usize,
        rng: &mut R,
    ) -> Result<(Vec<Experience>, Vec<f64>, Vec<usize>)> {
        if batch_size > self.buffer.len() {
            return Err(NumRs2Error::ValueError(format!(
                "Requested batch size {} exceeds buffer size {}",
                batch_size,
                self.buffer.len()
            )));
        }

        // Compute sampling probabilities
        let priorities_alpha: Vec<f64> = self
            .priorities
            .iter()
            .map(|&p| (p + self.epsilon).powf(self.alpha))
            .collect();

        let total_priority: f64 = priorities_alpha.iter().sum();
        let probabilities: Vec<f64> = priorities_alpha
            .iter()
            .map(|&p| p / total_priority)
            .collect();

        // Sample indices based on priorities using inverse transform sampling
        let mut indices = Vec::with_capacity(batch_size);
        let mut experiences = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        let uniform_dist = Uniform::new(0.0, 1.0)
            .map_err(|e| NumRs2Error::ValueError(format!("Uniform distribution error: {}", e)))?;

        for _ in 0..batch_size {
            // Inverse transform sampling
            let u = uniform_dist.sample(rng);
            let mut cumsum = 0.0;
            let mut idx = 0;

            for (i, &prob) in probabilities.iter().enumerate() {
                cumsum += prob;
                if u <= cumsum {
                    idx = i;
                    break;
                }
            }

            indices.push(idx);
            experiences.push(self.buffer[idx].clone());

            // Compute importance-sampling weight
            let prob = probabilities[idx];
            let weight = (self.buffer.len() as f64 * prob).powf(-self.beta);
            weights.push(weight);
        }

        // Normalize weights by max weight
        let max_weight = weights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let normalized_weights: Vec<f64> = weights.iter().map(|&w| w / max_weight).collect();

        // Increment beta
        self.beta = (self.beta + self.beta_increment).min(1.0);

        Ok((experiences, normalized_weights, indices))
    }

    /// Update priorities for sampled experiences based on TD-errors
    ///
    /// # Arguments
    /// * `indices` - Indices of experiences to update
    /// * `td_errors` - TD-errors for each experience (unsigned)
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f64]) -> Result<()> {
        if indices.len() != td_errors.len() {
            return Err(NumRs2Error::ValueError(
                "indices and td_errors must have same length".to_string(),
            ));
        }

        for (&idx, &td_error) in indices.iter().zip(td_errors.iter()) {
            if idx >= self.priorities.len() {
                return Err(NumRs2Error::ValueError(format!(
                    "Index {} out of bounds for buffer of size {}",
                    idx,
                    self.priorities.len()
                )));
            }

            let priority = td_error.abs() + self.epsilon;
            self.priorities[idx] = priority;
            self.max_priority = self.max_priority.max(priority);
        }

        Ok(())
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.priorities.clear();
        self.max_priority = 1.0;
    }

    /// Get current beta value
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Get alpha value
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::thread_rng;

    fn create_dummy_experience(value: f64) -> Experience {
        Experience {
            state: Array1::from_vec(vec![value]),
            action: 0,
            reward: value,
            next_state: Array1::from_vec(vec![value + 1.0]),
            done: false,
        }
    }

    #[test]
    fn test_experience_replay_creation() {
        let replay = ExperienceReplay::new(100);
        assert_eq!(replay.capacity(), 100);
        assert_eq!(replay.len(), 0);
        assert!(replay.is_empty());
    }

    #[test]
    #[should_panic(expected = "Capacity must be positive")]
    fn test_experience_replay_zero_capacity() {
        let _replay = ExperienceReplay::new(0);
    }

    #[test]
    fn test_experience_replay_push() {
        let mut replay = ExperienceReplay::new(3);
        replay.push(create_dummy_experience(1.0));
        assert_eq!(replay.len(), 1);

        replay.push(create_dummy_experience(2.0));
        replay.push(create_dummy_experience(3.0));
        assert_eq!(replay.len(), 3);
        assert!(replay.is_full());
    }

    #[test]
    fn test_experience_replay_overflow() {
        let mut replay = ExperienceReplay::new(2);
        replay.push(create_dummy_experience(1.0));
        replay.push(create_dummy_experience(2.0));
        replay.push(create_dummy_experience(3.0));

        assert_eq!(replay.len(), 2);
        // Should have removed first experience (value=1.0)
        let experiences = replay.experiences();
        assert_eq!(experiences[0].reward, 2.0);
        assert_eq!(experiences[1].reward, 3.0);
    }

    #[test]
    fn test_experience_replay_sample() -> Result<()> {
        let mut replay = ExperienceReplay::new(10);
        for i in 0..5 {
            replay.push(create_dummy_experience(i as f64));
        }

        let mut rng = thread_rng();
        let samples = replay.sample(3, &mut rng)?;
        assert_eq!(samples.len(), 3);
        Ok(())
    }

    #[test]
    fn test_experience_replay_sample_too_large() -> Result<()> {
        let mut replay = ExperienceReplay::new(10);
        replay.push(create_dummy_experience(1.0));

        let mut rng = thread_rng();
        let result = replay.sample(5, &mut rng);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_experience_replay_clear() {
        let mut replay = ExperienceReplay::new(10);
        replay.push(create_dummy_experience(1.0));
        replay.push(create_dummy_experience(2.0));
        replay.clear();

        assert_eq!(replay.len(), 0);
        assert!(replay.is_empty());
    }

    #[test]
    fn test_prioritized_replay_creation() -> Result<()> {
        let replay = PrioritizedExperienceReplay::new(100, 0.6, 0.4, 1e-6, 1e-6)?;
        assert_eq!(replay.capacity(), 100);
        assert_eq!(replay.len(), 0);
        assert_eq!(replay.alpha(), 0.6);
        assert_eq!(replay.beta(), 0.4);
        Ok(())
    }

    #[test]
    fn test_prioritized_replay_invalid_params() {
        assert!(PrioritizedExperienceReplay::new(0, 0.6, 0.4, 1e-6, 1e-6).is_err());
        assert!(PrioritizedExperienceReplay::new(100, -0.1, 0.4, 1e-6, 1e-6).is_err());
        assert!(PrioritizedExperienceReplay::new(100, 0.6, -0.1, 1e-6, 1e-6).is_err());
        assert!(PrioritizedExperienceReplay::new(100, 0.6, 1.5, 1e-6, 1e-6).is_err());
        assert!(PrioritizedExperienceReplay::new(100, 0.6, 0.4, 1e-6, 0.0).is_err());
    }

    #[test]
    fn test_prioritized_replay_push() -> Result<()> {
        let mut replay = PrioritizedExperienceReplay::new(3, 0.6, 0.4, 1e-6, 1e-6)?;
        replay.push(create_dummy_experience(1.0));
        assert_eq!(replay.len(), 1);

        replay.push(create_dummy_experience(2.0));
        replay.push(create_dummy_experience(3.0));
        assert_eq!(replay.len(), 3);
        Ok(())
    }

    #[test]
    fn test_prioritized_replay_overflow() -> Result<()> {
        let mut replay = PrioritizedExperienceReplay::new(2, 0.6, 0.4, 1e-6, 1e-6)?;
        replay.push(create_dummy_experience(1.0));
        replay.push(create_dummy_experience(2.0));
        replay.push(create_dummy_experience(3.0));

        assert_eq!(replay.len(), 2);
        Ok(())
    }

    #[test]
    fn test_prioritized_replay_sample() -> Result<()> {
        let mut replay = PrioritizedExperienceReplay::new(10, 0.6, 0.4, 1e-6, 1e-6)?;
        for i in 0..5 {
            replay.push(create_dummy_experience(i as f64));
        }

        let mut rng = thread_rng();
        let (experiences, weights, indices) = replay.sample(3, &mut rng)?;

        assert_eq!(experiences.len(), 3);
        assert_eq!(weights.len(), 3);
        assert_eq!(indices.len(), 3);

        // Check that weights are normalized (max should be 1.0)
        let max_weight = weights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!((max_weight - 1.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_prioritized_replay_update_priorities() -> Result<()> {
        let mut replay = PrioritizedExperienceReplay::new(10, 0.6, 0.4, 1e-6, 1e-6)?;
        for i in 0..5 {
            replay.push(create_dummy_experience(i as f64));
        }

        let mut rng = thread_rng();
        let (_, _, indices) = replay.sample(3, &mut rng)?;

        let td_errors = vec![1.0, 2.0, 0.5];
        replay.update_priorities(&indices, &td_errors)?;

        Ok(())
    }

    #[test]
    fn test_prioritized_replay_beta_increment() -> Result<()> {
        let mut replay = PrioritizedExperienceReplay::new(10, 0.6, 0.4, 0.1, 1e-6)?;
        for i in 0..5 {
            replay.push(create_dummy_experience(i as f64));
        }

        let initial_beta = replay.beta();
        let mut rng = thread_rng();

        for _ in 0..5 {
            let _ = replay.sample(2, &mut rng)?;
        }

        assert!(replay.beta() > initial_beta);
        assert!(replay.beta() <= 1.0);
        Ok(())
    }

    #[test]
    fn test_prioritized_replay_invalid_update() -> Result<()> {
        let mut replay = PrioritizedExperienceReplay::new(10, 0.6, 0.4, 1e-6, 1e-6)?;
        for i in 0..5 {
            replay.push(create_dummy_experience(i as f64));
        }

        // Mismatched lengths
        let result = replay.update_priorities(&[0, 1], &[1.0]);
        assert!(result.is_err());

        // Out of bounds index
        let result = replay.update_priorities(&[100], &[1.0]);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_prioritized_replay_clear() -> Result<()> {
        let mut replay = PrioritizedExperienceReplay::new(10, 0.6, 0.4, 1e-6, 1e-6)?;
        replay.push(create_dummy_experience(1.0));
        replay.push(create_dummy_experience(2.0));
        replay.clear();

        assert_eq!(replay.len(), 0);
        assert!(replay.is_empty());
        Ok(())
    }

    #[test]
    fn test_experience_clone() {
        let exp = create_dummy_experience(1.0);
        let cloned = exp.clone();

        assert_eq!(exp.state[0], cloned.state[0]);
        assert_eq!(exp.action, cloned.action);
        assert_eq!(exp.reward, cloned.reward);
        assert_eq!(exp.next_state[0], cloned.next_state[0]);
        assert_eq!(exp.done, cloned.done);
    }
}
