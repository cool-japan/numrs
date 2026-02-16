//! # Reinforcement Learning Module
//!
//! This module provides comprehensive reinforcement learning primitives for NumRS2,
//! including environment abstractions, various RL agents, experience replay buffers,
//! and exploration strategies.
//!
//! ## Overview
//!
//! The RL module offers production-ready implementations of:
//!
//! - **Environment Abstractions**: Trait-based interface compatible with OpenAI Gym
//! - **Classic Control Environments**: CartPole, MountainCar, Pendulum
//! - **RL Agents**: Q-Learning, SARSA, DQN, REINFORCE, Actor-Critic
//! - **Experience Replay**: Standard and prioritized experience replay
//! - **Exploration Strategies**: Epsilon-greedy, Boltzmann exploration
//! - **Utilities**: Reward normalization, episode tracking
//!
//! ## Mathematical Background
//!
//! ### Reinforcement Learning Framework
//!
//! RL agents learn optimal policies by interacting with environments. At each timestep t:
//! - Agent observes state s_t
//! - Agent selects action a_t according to policy π(a|s)
//! - Environment transitions to s_{t+1} and returns reward r_t
//! - Agent updates its policy/value function
//!
//! ### Q-Learning
//!
//! Q-Learning learns the optimal action-value function Q*(s,a) using the Bellman equation:
//!
//! ```text
//! Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
//! ```
//!
//! where α is the learning rate and γ is the discount factor.
//!
//! ### Deep Q-Network (DQN)
//!
//! DQN approximates Q(s,a) using a neural network θ:
//!
//! ```text
//! L(θ) = E[(r + γ max_a' Q(s',a';θ⁻) - Q(s,a;θ))²]
//! ```
//!
//! Uses experience replay and target networks for stability.
//!
//! ### Policy Gradient (REINFORCE)
//!
//! Directly optimizes the policy π_θ(a|s) by maximizing expected return:
//!
//! ```text
//! ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * G_t]
//! ```
//!
//! where G_t is the cumulative discounted return.
//!
//! ### Actor-Critic
//!
//! Combines value-based and policy-based methods:
//! - **Actor**: Updates policy π_θ(a|s)
//! - **Critic**: Estimates value function V_φ(s)
//!
//! ```text
//! Actor update: ∇_θ log π_θ(a|s) * A(s,a)
//! Critic update: minimize (V_φ(s) - G_t)²
//! ```
//!
//! where A(s,a) is the advantage function.
//!
//! ## SCIRS2 Policy Compliance
//!
//! This module strictly follows SCIRS2 ecosystem policies:
//!
//! - **Random Number Generation**: ALWAYS use `scirs2_core::random` (NEVER direct rand)
//! - **Array Operations**: ALWAYS use `scirs2_core::ndarray` (NEVER direct ndarray)
//! - **Parallel Processing**: ALWAYS use `scirs2_core::parallel_ops` (NEVER direct rayon)
//! - **Neural Networks**: Use NumRS2's `nn` module for DQN/Actor-Critic
//! - **Pure Rust**: 100% Pure Rust via SciRS2 ecosystem (no C/C++ dependencies)
//!
//! ## Usage Examples
//!
//! ### Example 1: Q-Learning on CartPole
//!
//! ```rust,ignore
//! use numrs2::new_modules::rl::*;
//! use scirs2_core::random::default_rng;
//!
//! let mut env = CartPoleEnv::new();
//! let mut agent = QLearningAgent::new(
//!     env.state_dim(),
//!     env.action_dim(),
//!     0.1,   // learning_rate
//!     0.99,  // gamma
//! )?;
//!
//! let mut rng = default_rng();
//! let mut exploration = EpsilonGreedy::new(1.0, 0.01, 0.995);
//!
//! for episode in 0..1000 {
//!     let mut state = env.reset(&mut rng)?;
//!     let mut total_reward = 0.0;
//!
//!     loop {
//!         let action = exploration.select_action(&mut agent, &state, &mut rng)?;
//!         let step = env.step(action, &mut rng)?;
//!
//!         agent.update(&state, action, step.reward, &step.next_state, step.done)?;
//!
//!         total_reward += step.reward;
//!         state = step.next_state;
//!
//!         if step.done {
//!             break;
//!         }
//!     }
//!
//!     exploration.decay();
//! }
//! ```
//!
//! ### Example 2: DQN with Experience Replay
//!
//! ```rust,ignore
//! use numrs2::new_modules::rl::*;
//! use scirs2_core::random::default_rng;
//!
//! let mut env = CartPoleEnv::new();
//! let mut agent = DQNAgent::new(
//!     env.state_dim(),
//!     env.action_dim(),
//!     vec![64, 64], // hidden_dims
//!     0.001,        // learning_rate
//!     0.99,         // gamma
//! )?;
//!
//! let mut replay_buffer = ExperienceReplay::new(10000);
//! let mut rng = default_rng();
//!
//! for episode in 0..500 {
//!     let mut state = env.reset(&mut rng)?;
//!
//!     loop {
//!         let action = agent.select_action(&state, 0.1, &mut rng)?;
//!         let step = env.step(action, &mut rng)?;
//!
//!         replay_buffer.push(Experience {
//!             state: state.clone(),
//!             action,
//!             reward: step.reward,
//!             next_state: step.next_state.clone(),
//!             done: step.done,
//!         });
//!
//!         if replay_buffer.len() >= 64 {
//!             let batch = replay_buffer.sample(64, &mut rng)?;
//!             agent.train_batch(&batch)?;
//!         }
//!
//!         state = step.next_state;
//!         if step.done {
//!             break;
//!         }
//!     }
//!
//!     if episode % 10 == 0 {
//!         agent.update_target_network()?;
//!     }
//! }
//! ```
//!
//! ## Quality Standards
//!
//! All code maintains strict quality standards:
//! - No `unwrap()` calls in production code
//! - Comprehensive error handling with `Result<T, NumRs2Error>`
//! - Full documentation with mathematical formulas and citations
//! - Extensive test coverage (>50 tests)
//! - SIMD optimization where applicable
//! - Numerical stability guarantees

pub mod agents;
pub mod environment;
pub mod replay;
pub mod utils;

// Re-export commonly used types
pub use agents::{
    ActorCriticAgent, DQNAgent, PolicyGradientAgent, QLearningAgent, RLAgent, SARSAAgent,
};
pub use environment::{CartPoleEnv, Environment, EnvironmentStep, MountainCarEnv, PendulumEnv};
pub use replay::{Experience, ExperienceReplay, PrioritizedExperienceReplay};
pub use utils::{
    BoltzmannExploration, EpisodeTracker, EpsilonGreedy, ExplorationStrategy, RewardNormalizer,
};
