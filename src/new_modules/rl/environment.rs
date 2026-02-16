//! Environment Abstractions for Reinforcement Learning
//!
//! This module provides trait-based environment abstractions compatible with OpenAI Gym,
//! along with implementations of classic control environments.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Distribution, Rng, Uniform};

/// Result of an environment step
#[derive(Debug, Clone)]
pub struct EnvironmentStep {
    /// Next state after taking action
    pub next_state: Array1<f64>,
    /// Reward received
    pub reward: f64,
    /// Whether episode is done
    pub done: bool,
}

/// Trait for RL environments
pub trait Environment {
    /// Get state space dimensionality
    fn state_dim(&self) -> usize;

    /// Get action space dimensionality (discrete actions)
    fn action_dim(&self) -> usize;

    /// Reset environment and return initial state
    fn reset<R: Rng>(&mut self, rng: &mut R) -> Result<Array1<f64>>;

    /// Take action and return step result
    fn step<R: Rng>(&mut self, action: usize, rng: &mut R) -> Result<EnvironmentStep>;

    /// Check if state is terminal
    fn is_terminal(&self, state: &Array1<f64>) -> bool;

    /// Get observation bounds (optional)
    fn observation_bounds(&self) -> Option<(Array1<f64>, Array1<f64>)> {
        None
    }
}

/// CartPole environment
///
/// Classic control problem where an agent must balance a pole on a cart.
///
/// # State Space
/// - Position: [-4.8, 4.8]
/// - Velocity: [-∞, ∞]
/// - Angle: [-24°, 24°]
/// - Angular velocity: [-∞, ∞]
///
/// # Action Space
/// - 0: Push left
/// - 1: Push right
///
/// # Rewards
/// - +1 for every timestep the pole remains upright
///
/// # Episode Termination
/// - Pole angle exceeds ±12°
/// - Cart position exceeds ±2.4
/// - Episode length exceeds 500 steps
pub struct CartPoleEnv {
    state: Array1<f64>,
    steps: usize,
    gravity: f64,
    mass_cart: f64,
    mass_pole: f64,
    total_mass: f64,
    length: f64,
    pole_mass_length: f64,
    force_mag: f64,
    tau: f64,
    max_steps: usize,
}

impl CartPoleEnv {
    /// Create new CartPole environment with default parameters
    pub fn new() -> Self {
        let gravity = 9.8;
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        let total_mass = mass_cart + mass_pole;
        let length = 0.5;
        let pole_mass_length = mass_pole * length;
        let force_mag = 10.0;
        let tau = 0.02;

        Self {
            state: Array1::zeros(4),
            steps: 0,
            gravity,
            mass_cart,
            mass_pole,
            total_mass,
            length,
            pole_mass_length,
            force_mag,
            tau,
            max_steps: 500,
        }
    }

    /// Create CartPole with custom parameters
    pub fn with_params(
        gravity: f64,
        mass_cart: f64,
        mass_pole: f64,
        length: f64,
        force_mag: f64,
        tau: f64,
        max_steps: usize,
    ) -> Self {
        let total_mass = mass_cart + mass_pole;
        let pole_mass_length = mass_pole * length;

        Self {
            state: Array1::zeros(4),
            steps: 0,
            gravity,
            mass_cart,
            mass_pole,
            total_mass,
            length,
            pole_mass_length,
            force_mag,
            tau,
            max_steps,
        }
    }
}

impl Default for CartPoleEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for CartPoleEnv {
    fn state_dim(&self) -> usize {
        4
    }

    fn action_dim(&self) -> usize {
        2
    }

    fn reset<R: Rng>(&mut self, rng: &mut R) -> Result<Array1<f64>> {
        let dist = Uniform::new(-0.05, 0.05)
            .map_err(|e| NumRs2Error::InvalidInput(format!("Uniform distribution error: {}", e)))?;

        self.state = Array1::from_vec(vec![
            dist.sample(rng),
            dist.sample(rng),
            dist.sample(rng),
            dist.sample(rng),
        ]);
        self.steps = 0;
        Ok(self.state.clone())
    }

    fn step<R: Rng>(&mut self, action: usize, _rng: &mut R) -> Result<EnvironmentStep> {
        if action >= 2 {
            return Err(NumRs2Error::InvalidInput(format!(
                "Invalid action: {}. CartPole has 2 actions.",
                action
            )));
        }

        let x = self.state[0];
        let x_dot = self.state[1];
        let theta = self.state[2];
        let theta_dot = self.state[3];

        let force = if action == 1 {
            self.force_mag
        } else {
            -self.force_mag
        };

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let temp =
            (force + self.pole_mass_length * theta_dot * theta_dot * sin_theta) / self.total_mass;
        let theta_acc = (self.gravity * sin_theta - cos_theta * temp)
            / (self.length
                * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass));
        let x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass;

        // Update state
        self.state[0] = x + self.tau * x_dot;
        self.state[1] = x_dot + self.tau * x_acc;
        self.state[2] = theta + self.tau * theta_dot;
        self.state[3] = theta_dot + self.tau * theta_acc;

        self.steps += 1;

        let done = self.is_terminal(&self.state) || self.steps >= self.max_steps;
        let reward = if done { 0.0 } else { 1.0 };

        Ok(EnvironmentStep {
            next_state: self.state.clone(),
            reward,
            done,
        })
    }

    fn is_terminal(&self, state: &Array1<f64>) -> bool {
        let x = state[0];
        let theta = state[2];
        x.abs() > 2.4 || theta.abs() > 0.2095 // ~12 degrees
    }

    fn observation_bounds(&self) -> Option<(Array1<f64>, Array1<f64>)> {
        let low = Array1::from_vec(vec![-4.8, f64::NEG_INFINITY, -0.418, f64::NEG_INFINITY]);
        let high = Array1::from_vec(vec![4.8, f64::INFINITY, 0.418, f64::INFINITY]);
        Some((low, high))
    }
}

/// Mountain Car environment
///
/// An underpowered car must drive up a steep mountain by building momentum.
///
/// # State Space
/// - Position: [-1.2, 0.6]
/// - Velocity: [-0.07, 0.07]
///
/// # Action Space
/// - 0: Accelerate left
/// - 1: Don't accelerate
/// - 2: Accelerate right
///
/// # Rewards
/// - -1 for every timestep until goal is reached
///
/// # Episode Termination
/// - Car reaches position >= 0.5
/// - Episode length exceeds 200 steps
pub struct MountainCarEnv {
    state: Array1<f64>,
    steps: usize,
    min_position: f64,
    max_position: f64,
    max_speed: f64,
    goal_position: f64,
    force: f64,
    gravity: f64,
    max_steps: usize,
}

impl MountainCarEnv {
    /// Create new MountainCar environment with default parameters
    pub fn new() -> Self {
        Self {
            state: Array1::zeros(2),
            steps: 0,
            min_position: -1.2,
            max_position: 0.6,
            max_speed: 0.07,
            goal_position: 0.5,
            force: 0.001,
            gravity: 0.0025,
            max_steps: 200,
        }
    }

    /// Create MountainCar with custom parameters
    pub fn with_params(
        min_position: f64,
        max_position: f64,
        max_speed: f64,
        goal_position: f64,
        force: f64,
        gravity: f64,
        max_steps: usize,
    ) -> Self {
        Self {
            state: Array1::zeros(2),
            steps: 0,
            min_position,
            max_position,
            max_speed,
            goal_position,
            force,
            gravity,
            max_steps,
        }
    }
}

impl Default for MountainCarEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for MountainCarEnv {
    fn state_dim(&self) -> usize {
        2
    }

    fn action_dim(&self) -> usize {
        3
    }

    fn reset<R: Rng>(&mut self, rng: &mut R) -> Result<Array1<f64>> {
        let dist = Uniform::new(-0.6, -0.4)
            .map_err(|e| NumRs2Error::InvalidInput(format!("Uniform distribution error: {}", e)))?;

        let position = dist.sample(rng);
        self.state = Array1::from_vec(vec![position, 0.0]);
        self.steps = 0;
        Ok(self.state.clone())
    }

    fn step<R: Rng>(&mut self, action: usize, _rng: &mut R) -> Result<EnvironmentStep> {
        if action >= 3 {
            return Err(NumRs2Error::InvalidInput(format!(
                "Invalid action: {}. MountainCar has 3 actions.",
                action
            )));
        }

        let position = self.state[0];
        let velocity = self.state[1];

        let force = match action {
            0 => -self.force,
            1 => 0.0,
            2 => self.force,
            _ => unreachable!(),
        };

        let new_velocity = velocity + force - self.gravity * (3.0 * position).cos();
        let new_velocity = new_velocity.clamp(-self.max_speed, self.max_speed);

        let new_position = position + new_velocity;
        let new_position = new_position.clamp(self.min_position, self.max_position);

        // Reset velocity if hit left boundary
        let new_velocity = if new_position == self.min_position && new_velocity < 0.0 {
            0.0
        } else {
            new_velocity
        };

        self.state[0] = new_position;
        self.state[1] = new_velocity;
        self.steps += 1;

        let done = new_position >= self.goal_position || self.steps >= self.max_steps;
        let reward = if new_position >= self.goal_position {
            0.0
        } else {
            -1.0
        };

        Ok(EnvironmentStep {
            next_state: self.state.clone(),
            reward,
            done,
        })
    }

    fn is_terminal(&self, state: &Array1<f64>) -> bool {
        state[0] >= self.goal_position
    }

    fn observation_bounds(&self) -> Option<(Array1<f64>, Array1<f64>)> {
        let low = Array1::from_vec(vec![self.min_position, -self.max_speed]);
        let high = Array1::from_vec(vec![self.max_position, self.max_speed]);
        Some((low, high))
    }
}

/// Pendulum environment
///
/// Classic control problem of swinging up and balancing an inverted pendulum.
///
/// # State Space
/// - cos(theta): [-1.0, 1.0]
/// - sin(theta): [-1.0, 1.0]
/// - Angular velocity: [-8.0, 8.0]
///
/// # Action Space
/// - Continuous torque in [-2.0, 2.0]
/// - Discretized into bins for discrete action space
///
/// # Rewards
/// - -(theta² + 0.1*theta_dot² + 0.001*action²)
/// - Closer to upright position = higher reward
///
/// # Episode Termination
/// - Episode length exceeds 200 steps
pub struct PendulumEnv {
    state: Array1<f64>,
    steps: usize,
    max_speed: f64,
    max_torque: f64,
    dt: f64,
    g: f64,
    m: f64,
    l: f64,
    max_steps: usize,
    action_bins: usize,
}

impl PendulumEnv {
    /// Create new Pendulum environment with default parameters
    pub fn new() -> Self {
        Self::with_action_bins(5)
    }

    /// Create Pendulum with specific number of discrete action bins
    pub fn with_action_bins(action_bins: usize) -> Self {
        Self {
            state: Array1::zeros(3),
            steps: 0,
            max_speed: 8.0,
            max_torque: 2.0,
            dt: 0.05,
            g: 10.0,
            m: 1.0,
            l: 1.0,
            max_steps: 200,
            action_bins,
        }
    }

    /// Create Pendulum with custom parameters
    pub fn with_params(
        max_speed: f64,
        max_torque: f64,
        dt: f64,
        g: f64,
        m: f64,
        l: f64,
        max_steps: usize,
        action_bins: usize,
    ) -> Self {
        Self {
            state: Array1::zeros(3),
            steps: 0,
            max_speed,
            max_torque,
            dt,
            g,
            m,
            l,
            max_steps,
            action_bins,
        }
    }

    /// Convert discrete action to continuous torque
    fn action_to_torque(&self, action: usize) -> f64 {
        let step = 2.0 * self.max_torque / (self.action_bins - 1) as f64;
        -self.max_torque + action as f64 * step
    }

    /// Normalize angle to [-pi, pi]
    fn angle_normalize(angle: f64) -> f64 {
        let two_pi = 2.0 * std::f64::consts::PI;
        let normalized = ((angle + std::f64::consts::PI) % two_pi + two_pi) % two_pi;
        normalized - std::f64::consts::PI
    }
}

impl Default for PendulumEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for PendulumEnv {
    fn state_dim(&self) -> usize {
        3 // cos(theta), sin(theta), theta_dot
    }

    fn action_dim(&self) -> usize {
        self.action_bins
    }

    fn reset<R: Rng>(&mut self, rng: &mut R) -> Result<Array1<f64>> {
        let dist = Uniform::new(-std::f64::consts::PI, std::f64::consts::PI)
            .map_err(|e| NumRs2Error::InvalidInput(format!("Uniform distribution error: {}", e)))?;
        let vel_dist = Uniform::new(-1.0, 1.0)
            .map_err(|e| NumRs2Error::InvalidInput(format!("Uniform distribution error: {}", e)))?;

        let theta = dist.sample(rng);
        let theta_dot = vel_dist.sample(rng);

        self.state = Array1::from_vec(vec![theta.cos(), theta.sin(), theta_dot]);
        self.steps = 0;
        Ok(self.state.clone())
    }

    fn step<R: Rng>(&mut self, action: usize, _rng: &mut R) -> Result<EnvironmentStep> {
        if action >= self.action_bins {
            return Err(NumRs2Error::InvalidInput(format!(
                "Invalid action: {}. Pendulum has {} actions.",
                action, self.action_bins
            )));
        }

        let cos_theta = self.state[0];
        let sin_theta = self.state[1];
        let theta_dot = self.state[2];

        let theta = sin_theta.atan2(cos_theta);
        let u = self
            .action_to_torque(action)
            .clamp(-self.max_torque, self.max_torque);

        // Compute costs (negative reward)
        let costs = theta * theta + 0.1 * theta_dot * theta_dot + 0.001 * u * u;

        // Update dynamics
        let new_theta_dot = theta_dot
            + (3.0 * self.g / (2.0 * self.l) * theta.sin() + 3.0 / (self.m * self.l * self.l) * u)
                * self.dt;
        let new_theta_dot = new_theta_dot.clamp(-self.max_speed, self.max_speed);

        let new_theta = Self::angle_normalize(theta + new_theta_dot * self.dt);

        self.state[0] = new_theta.cos();
        self.state[1] = new_theta.sin();
        self.state[2] = new_theta_dot;
        self.steps += 1;

        let done = self.steps >= self.max_steps;
        let reward = -costs;

        Ok(EnvironmentStep {
            next_state: self.state.clone(),
            reward,
            done,
        })
    }

    fn is_terminal(&self, _state: &Array1<f64>) -> bool {
        false // Pendulum is never terminal except by max_steps
    }

    fn observation_bounds(&self) -> Option<(Array1<f64>, Array1<f64>)> {
        let low = Array1::from_vec(vec![-1.0, -1.0, -self.max_speed]);
        let high = Array1::from_vec(vec![1.0, 1.0, self.max_speed]);
        Some((low, high))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_cartpole_creation() -> Result<()> {
        let env = CartPoleEnv::new();
        assert_eq!(env.state_dim(), 4);
        assert_eq!(env.action_dim(), 2);
        Ok(())
    }

    #[test]
    fn test_cartpole_reset() -> Result<()> {
        let mut env = CartPoleEnv::new();
        let mut rng = thread_rng();
        let state = env.reset(&mut rng)?;
        assert_eq!(state.len(), 4);
        assert!(state[0].abs() <= 0.05);
        assert!(state[1].abs() <= 0.05);
        assert!(state[2].abs() <= 0.05);
        assert!(state[3].abs() <= 0.05);
        Ok(())
    }

    #[test]
    fn test_cartpole_step() -> Result<()> {
        let mut env = CartPoleEnv::new();
        let mut rng = thread_rng();
        env.reset(&mut rng)?;

        let step_result = env.step(0, &mut rng)?;
        assert_eq!(step_result.next_state.len(), 4);
        assert!(step_result.reward == 0.0 || step_result.reward == 1.0);
        Ok(())
    }

    #[test]
    fn test_cartpole_invalid_action() -> Result<()> {
        let mut env = CartPoleEnv::new();
        let mut rng = thread_rng();
        env.reset(&mut rng)?;

        let result = env.step(5, &mut rng);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_cartpole_terminal_state() -> Result<()> {
        let env = CartPoleEnv::new();
        let terminal_state = Array1::from_vec(vec![3.0, 0.0, 0.0, 0.0]); // x > 2.4
        assert!(env.is_terminal(&terminal_state));

        let non_terminal_state = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        assert!(!env.is_terminal(&non_terminal_state));
        Ok(())
    }

    #[test]
    fn test_mountaincar_creation() -> Result<()> {
        let env = MountainCarEnv::new();
        assert_eq!(env.state_dim(), 2);
        assert_eq!(env.action_dim(), 3);
        Ok(())
    }

    #[test]
    fn test_mountaincar_reset() -> Result<()> {
        let mut env = MountainCarEnv::new();
        let mut rng = thread_rng();
        let state = env.reset(&mut rng)?;
        assert_eq!(state.len(), 2);
        assert!(state[0] >= -0.6 && state[0] <= -0.4);
        assert_eq!(state[1], 0.0);
        Ok(())
    }

    #[test]
    fn test_mountaincar_step() -> Result<()> {
        let mut env = MountainCarEnv::new();
        let mut rng = thread_rng();
        env.reset(&mut rng)?;

        let step_result = env.step(2, &mut rng)?;
        assert_eq!(step_result.next_state.len(), 2);
        assert!(step_result.reward <= 0.0);
        Ok(())
    }

    #[test]
    fn test_mountaincar_invalid_action() -> Result<()> {
        let mut env = MountainCarEnv::new();
        let mut rng = thread_rng();
        env.reset(&mut rng)?;

        let result = env.step(5, &mut rng);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_mountaincar_goal_reached() -> Result<()> {
        let mut env = MountainCarEnv::new();
        env.state = Array1::from_vec(vec![0.5, 0.05]);

        let mut rng = thread_rng();
        let step_result = env.step(2, &mut rng)?;
        assert!(step_result.done);
        assert_eq!(step_result.reward, 0.0);
        Ok(())
    }

    #[test]
    fn test_pendulum_creation() -> Result<()> {
        let env = PendulumEnv::new();
        assert_eq!(env.state_dim(), 3);
        assert_eq!(env.action_dim(), 5);
        Ok(())
    }

    #[test]
    fn test_pendulum_custom_bins() -> Result<()> {
        let env = PendulumEnv::with_action_bins(7);
        assert_eq!(env.action_dim(), 7);
        Ok(())
    }

    #[test]
    fn test_pendulum_reset() -> Result<()> {
        let mut env = PendulumEnv::new();
        let mut rng = thread_rng();
        let state = env.reset(&mut rng)?;
        assert_eq!(state.len(), 3);
        // cos^2 + sin^2 = 1
        assert!((state[0] * state[0] + state[1] * state[1] - 1.0).abs() < 1e-6);
        assert!(state[2].abs() <= 1.0);
        Ok(())
    }

    #[test]
    fn test_pendulum_step() -> Result<()> {
        let mut env = PendulumEnv::new();
        let mut rng = thread_rng();
        env.reset(&mut rng)?;

        let step_result = env.step(2, &mut rng)?;
        assert_eq!(step_result.next_state.len(), 3);
        assert!(step_result.reward <= 0.0); // Pendulum rewards are always negative
        Ok(())
    }

    #[test]
    fn test_pendulum_invalid_action() -> Result<()> {
        let mut env = PendulumEnv::new();
        let mut rng = thread_rng();
        env.reset(&mut rng)?;

        let result = env.step(10, &mut rng);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_pendulum_angle_normalization() -> Result<()> {
        // π + 0.1 should normalize to -π + 0.1 (wrapping around)
        let angle1 = PendulumEnv::angle_normalize(std::f64::consts::PI + 0.1);
        assert!((angle1 - (-std::f64::consts::PI + 0.1)).abs() < 1e-6);

        // -π - 0.1 should normalize to π - 0.1 (wrapping around)
        let angle2 = PendulumEnv::angle_normalize(-std::f64::consts::PI - 0.1);
        assert!((angle2 - (std::f64::consts::PI - 0.1)).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_environment_observation_bounds() -> Result<()> {
        let cartpole = CartPoleEnv::new();
        let bounds = cartpole.observation_bounds();
        assert!(bounds.is_some());

        let (low, high) =
            bounds.ok_or_else(|| NumRs2Error::InvalidInput("Bounds should exist".to_string()))?;
        assert_eq!(low.len(), 4);
        assert_eq!(high.len(), 4);
        Ok(())
    }

    #[test]
    fn test_cartpole_episode_length() -> Result<()> {
        let mut env = CartPoleEnv::new();
        let mut rng = thread_rng();
        env.reset(&mut rng)?;

        // Step until episode ends
        let mut steps = 0;
        loop {
            let step_result = env.step(0, &mut rng)?;
            steps += 1;
            if step_result.done {
                break;
            }
            if steps > 600 {
                break;
            }
        }

        assert!(steps <= 500);
        Ok(())
    }
}
