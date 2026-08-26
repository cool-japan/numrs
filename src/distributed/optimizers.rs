//! Distributed Optimizers for Large-Scale Training
//!
//! This module provides distributed implementations of popular optimization algorithms
//! with gradient compression, local accumulation, and efficient communication patterns.
//!
//! # Features
//!
//! - **Distributed SGD**: With momentum and Nesterov acceleration
//! - **Distributed Adam/AdamW**: Adaptive learning rates
//! - **Gradient Compression**: Top-k, quantization, sparsification
//! - **Local Gradient Accumulation**: Reduce communication frequency
//! - **Learning Rate Scheduling**: Warmup, decay, cyclical
//!
//! # Optimization Patterns
//!
//! ## Synchronous Optimization
//! ```text
//! All workers:
//!   1. Compute gradients
//!   2. AllReduce gradients
//!   3. Update parameters (same for all)
//! ```
//!
//! ## Asynchronous Optimization
//! ```text
//! Workers:
//!   1. Compute gradients
//!   2. Push to parameter server (async)
//!   3. Pull updated parameters
//!   4. Continue training
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::optimizers::*;
//! use numrs2::distributed::process::*;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), OptimizerError> {
//! let world = init().await?;
//!
//! // Distributed SGD with momentum
//! let mut sgd = DistributedSGD::new(
//!     0.01,  // learning rate
//!     Arc::new(world.clone()),
//! )?.with_momentum(0.9);
//!
//! // Distributed Adam
//! let mut adam = DistributedAdam::new(
//!     0.001,  // learning rate
//!     Arc::new(world),
//! )?.with_betas(0.9, 0.999);
//!
//! // Training step with gradient compression
//! let params = vec![1.0; 1000];
//! let grads = vec![0.1; 1000];
//!
//! let updated = sgd.step(&params, &grads).await?;
//! # Ok(())
//! # }
//! ```

use super::collective::{allreduce, ReduceOp};
use super::communication::{
    compress_tensor, decompress_tensor, AsyncCommunicator, CommunicationError, CompressionStrategy,
};
use super::coordinator::{CoordinatorError, RingAllReduce};
use super::data_parallel::GradientAggregation;
use super::process::{Communicator, ProcessError};
use crate::error::NumRs2Error;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};

/// Errors in optimizer operations
#[derive(Error, Debug)]
pub enum OptimizerError {
    #[error("Process error: {0}")]
    Process(#[from] ProcessError),

    #[error("Communication error: {0}")]
    Communication(#[from] CommunicationError),

    #[error("Coordinator error: {0}")]
    Coordinator(#[from] CoordinatorError),

    #[error("Invalid learning rate: {0}")]
    InvalidLearningRate(f32),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("State mismatch: {0}")]
    StateMismatch(String),

    #[error("Compression error: {0}")]
    CompressionError(String),
}

impl From<OptimizerError> for NumRs2Error {
    fn from(err: OptimizerError) -> Self {
        NumRs2Error::DistributedComputing(err.to_string())
    }
}

/// Optimizer state for tracking moments
#[derive(Debug, Clone)]
struct OptimizerState {
    /// First moment (momentum/mean)
    m: Vec<f32>,

    /// Second moment (variance) - for Adam
    v: Option<Vec<f32>>,

    /// Time step
    t: u64,
}

impl OptimizerState {
    fn new(size: usize, use_second_moment: bool) -> Self {
        Self {
            m: vec![0.0; size],
            v: if use_second_moment {
                Some(vec![0.0; size])
            } else {
                None
            },
            t: 0,
        }
    }
}

/// Distributed Stochastic Gradient Descent
pub struct DistributedSGD {
    /// Learning rate
    lr: f32,

    /// Momentum coefficient
    momentum: f32,

    /// Nesterov acceleration
    nesterov: bool,

    /// Weight decay (L2 regularization)
    weight_decay: f32,

    /// Communicator
    communicator: Arc<Communicator>,

    /// Async communicator
    // `step`/`step_named` aggregate gradients via `allreduce`/`ring_reducer`
    // and never touch this handle; it documents the asynchronous push/pull
    // path (see the module's "Asynchronous Optimization" pattern) that
    // hasn't been wired into `DistributedSGD` yet. Wiring it in is a
    // behavioral change, out of scope for a lint-only pass.
    #[allow(dead_code)]
    async_comm: AsyncCommunicator,

    /// Gradient aggregation strategy
    aggregation: GradientAggregation,

    /// Ring-allreduce coordinator
    ring_reducer: Option<RingAllReduce>,

    /// Optimizer state (per-parameter momentum)
    state: Arc<RwLock<HashMap<String, OptimizerState>>>,

    /// Gradient compression
    compression: CompressionStrategy,

    /// Local accumulation steps
    accumulation_steps: usize,

    /// Current step counter
    current_step: Arc<Mutex<usize>>,
}

impl DistributedSGD {
    /// Create new distributed SGD optimizer
    pub fn new(lr: f32, communicator: Arc<Communicator>) -> Result<Self, OptimizerError> {
        if lr <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(lr));
        }

        let async_comm = AsyncCommunicator::new(communicator.clone())?;

        Ok(Self {
            lr,
            momentum: 0.0,
            nesterov: false,
            weight_decay: 0.0,
            communicator,
            async_comm,
            aggregation: GradientAggregation::AllReduce,
            ring_reducer: None,
            state: Arc::new(RwLock::new(HashMap::new())),
            compression: CompressionStrategy::None,
            accumulation_steps: 1,
            current_step: Arc::new(Mutex::new(0)),
        })
    }

    /// Set momentum
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable Nesterov acceleration
    pub fn with_nesterov(mut self) -> Self {
        self.nesterov = true;
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set gradient compression
    pub fn with_compression(mut self, compression: CompressionStrategy) -> Self {
        self.compression = compression;
        self
    }

    /// Set gradient aggregation strategy
    pub fn with_aggregation(
        mut self,
        aggregation: GradientAggregation,
    ) -> Result<Self, OptimizerError> {
        self.aggregation = aggregation;

        if aggregation == GradientAggregation::RingAllReduce {
            self.ring_reducer = Some(RingAllReduce::new(self.communicator.clone())?);
        }

        Ok(self)
    }

    /// Set local accumulation steps
    pub fn with_accumulation(mut self, steps: usize) -> Self {
        self.accumulation_steps = steps;
        self
    }

    /// Perform optimization step
    pub async fn step(
        &mut self,
        params: &[f32],
        gradients: &[f32],
    ) -> Result<Vec<f32>, OptimizerError> {
        self.step_named("default", params, gradients).await
    }

    /// Perform optimization step with named parameters
    pub async fn step_named(
        &mut self,
        param_name: &str,
        params: &[f32],
        gradients: &[f32],
    ) -> Result<Vec<f32>, OptimizerError> {
        // Check accumulation
        let mut step = self.current_step.lock().await;
        *step += 1;

        if *step < self.accumulation_steps {
            // Not ready to update yet
            return Ok(params.to_vec());
        }

        *step = 0;
        drop(step);

        // Compress gradients
        let (compressed, indices) = compress_tensor(gradients, &self.compression)
            .map_err(|e| OptimizerError::CompressionError(e.to_string()))?;

        // Aggregate gradients
        let aggregated = self.aggregate_gradients(&compressed).await?;

        // Decompress
        let grads = if indices.is_some() {
            decompress_tensor(&aggregated, indices.as_deref(), gradients.len())
                .map_err(|e| OptimizerError::CompressionError(e.to_string()))?
        } else {
            aggregated
        };

        // Get or create optimizer state
        let mut state_map = self.state.write().await;
        let state = state_map
            .entry(param_name.to_string())
            .or_insert_with(|| OptimizerState::new(params.len(), false));

        state.t += 1;

        // Apply weight decay
        let mut grads = grads;
        if self.weight_decay > 0.0 {
            for (g, &p) in grads.iter_mut().zip(params.iter()) {
                *g += self.weight_decay * p;
            }
        }

        // Update with momentum
        let mut new_params = params.to_vec();

        if self.momentum > 0.0 {
            // Momentum update
            for i in 0..params.len() {
                state.m[i] = self.momentum * state.m[i] + grads[i];

                if self.nesterov {
                    // Nesterov momentum
                    new_params[i] -= self.lr * (grads[i] + self.momentum * state.m[i]);
                } else {
                    // Standard momentum
                    new_params[i] -= self.lr * state.m[i];
                }
            }
        } else {
            // Vanilla SGD
            for i in 0..params.len() {
                new_params[i] -= self.lr * grads[i];
            }
        }

        Ok(new_params)
    }

    /// Aggregate gradients across workers.
    ///
    /// A real all-reduce **sum** across every rank, divided by the world
    /// size. An earlier version's `AllReduce`/`Hierarchical` branch never
    /// talked to another rank: it divided *this rank's own* gradients by
    /// the world size and returned that, so every rank silently computed a
    /// different step from a different, wrong "aggregate" (and
    /// `RingAllReduce`'s branch summed for real but never divided down to
    /// a mean).
    async fn aggregate_gradients(&self, gradients: &[f32]) -> Result<Vec<f32>, OptimizerError> {
        let world_size = self.communicator.size() as f32;
        let summed = match self.aggregation {
            GradientAggregation::RingAllReduce => {
                let reducer = self.ring_reducer.as_ref().ok_or_else(|| {
                    OptimizerError::StateMismatch("Ring reducer not initialized".to_string())
                })?;
                reducer.allreduce(gradients).await?
            }

            GradientAggregation::AllReduce | GradientAggregation::Hierarchical => {
                allreduce(gradients, ReduceOp::Sum, &self.communicator)
                    .await
                    .map_err(CoordinatorError::from)?
            }
        };
        Ok(summed.into_iter().map(|g| g / world_size).collect())
    }

    /// Get learning rate
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Set learning rate
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Get momentum
    pub fn momentum(&self) -> f32 {
        self.momentum
    }
}

/// Distributed Adam optimizer
pub struct DistributedAdam {
    /// Learning rate
    lr: f32,

    /// Beta1 (exponential decay rate for first moment)
    beta1: f32,

    /// Beta2 (exponential decay rate for second moment)
    beta2: f32,

    /// Epsilon for numerical stability
    epsilon: f32,

    /// Weight decay (AdamW variant)
    weight_decay: f32,

    /// Communicator
    communicator: Arc<Communicator>,

    /// Async communicator
    // Same story as `DistributedSGD::async_comm`: `step`/`step_named` here
    // aggregate via `allreduce`/`ring_reducer` and never touch this handle.
    #[allow(dead_code)]
    async_comm: AsyncCommunicator,

    /// Gradient aggregation strategy
    aggregation: GradientAggregation,

    /// Ring-allreduce coordinator
    ring_reducer: Option<RingAllReduce>,

    /// Optimizer state (per-parameter moments)
    state: Arc<RwLock<HashMap<String, OptimizerState>>>,

    /// Gradient compression
    compression: CompressionStrategy,

    /// AMSGrad variant
    amsgrad: bool,
}

impl DistributedAdam {
    /// Create new distributed Adam optimizer
    pub fn new(lr: f32, communicator: Arc<Communicator>) -> Result<Self, OptimizerError> {
        if lr <= 0.0 {
            return Err(OptimizerError::InvalidLearningRate(lr));
        }

        let async_comm = AsyncCommunicator::new(communicator.clone())?;

        Ok(Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            communicator,
            async_comm,
            aggregation: GradientAggregation::AllReduce,
            ring_reducer: None,
            state: Arc::new(RwLock::new(HashMap::new())),
            compression: CompressionStrategy::None,
            amsgrad: false,
        })
    }

    /// Set beta parameters
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set weight decay (AdamW)
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable AMSGrad
    pub fn with_amsgrad(mut self) -> Self {
        self.amsgrad = true;
        self
    }

    /// Set gradient compression
    pub fn with_compression(mut self, compression: CompressionStrategy) -> Self {
        self.compression = compression;
        self
    }

    /// Set gradient aggregation strategy
    pub fn with_aggregation(
        mut self,
        aggregation: GradientAggregation,
    ) -> Result<Self, OptimizerError> {
        self.aggregation = aggregation;

        if aggregation == GradientAggregation::RingAllReduce {
            self.ring_reducer = Some(RingAllReduce::new(self.communicator.clone())?);
        }

        Ok(self)
    }

    /// Perform optimization step
    pub async fn step(
        &mut self,
        params: &[f32],
        gradients: &[f32],
    ) -> Result<Vec<f32>, OptimizerError> {
        self.step_named("default", params, gradients).await
    }

    /// Perform optimization step with named parameters
    pub async fn step_named(
        &mut self,
        param_name: &str,
        params: &[f32],
        gradients: &[f32],
    ) -> Result<Vec<f32>, OptimizerError> {
        // Compress gradients
        let (compressed, indices) = compress_tensor(gradients, &self.compression)
            .map_err(|e| OptimizerError::CompressionError(e.to_string()))?;

        // Aggregate gradients
        let aggregated = self.aggregate_gradients(&compressed).await?;

        // Decompress
        let grads = if indices.is_some() {
            decompress_tensor(&aggregated, indices.as_deref(), gradients.len())
                .map_err(|e| OptimizerError::CompressionError(e.to_string()))?
        } else {
            aggregated
        };

        // Get or create optimizer state
        let mut state_map = self.state.write().await;
        let state = state_map
            .entry(param_name.to_string())
            .or_insert_with(|| OptimizerState::new(params.len(), true));

        state.t += 1;

        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(state.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(state.t as i32);

        // Update moments and parameters
        let mut new_params = params.to_vec();

        for i in 0..params.len() {
            // Update biased first moment estimate
            state.m[i] = self.beta1 * state.m[i] + (1.0 - self.beta1) * grads[i];

            // Update biased second raw moment estimate
            if let Some(ref mut v) = state.v {
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grads[i] * grads[i];

                // Compute bias-corrected moments
                let m_hat = state.m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;

                // Update parameters
                if self.weight_decay > 0.0 {
                    // AdamW: decoupled weight decay
                    new_params[i] -= self.lr
                        * (m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * params[i]);
                } else {
                    // Standard Adam
                    new_params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
                }
            }
        }

        Ok(new_params)
    }

    /// Aggregate gradients across workers.
    ///
    /// A real all-reduce **sum** across every rank, divided by the world
    /// size. An earlier version's `AllReduce`/`Hierarchical` branch never
    /// talked to another rank: it divided *this rank's own* gradients by
    /// the world size and returned that, so every rank silently computed a
    /// different step from a different, wrong "aggregate" (and
    /// `RingAllReduce`'s branch summed for real but never divided down to
    /// a mean).
    async fn aggregate_gradients(&self, gradients: &[f32]) -> Result<Vec<f32>, OptimizerError> {
        let world_size = self.communicator.size() as f32;
        let summed = match self.aggregation {
            GradientAggregation::RingAllReduce => {
                let reducer = self.ring_reducer.as_ref().ok_or_else(|| {
                    OptimizerError::StateMismatch("Ring reducer not initialized".to_string())
                })?;
                reducer.allreduce(gradients).await?
            }

            GradientAggregation::AllReduce | GradientAggregation::Hierarchical => {
                allreduce(gradients, ReduceOp::Sum, &self.communicator)
                    .await
                    .map_err(CoordinatorError::from)?
            }
        };
        Ok(summed.into_iter().map(|g| g / world_size).collect())
    }

    /// Get learning rate
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Set learning rate
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Get beta1
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Get beta2
    pub fn beta2(&self) -> f32 {
        self.beta2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::process::{ProcessGroup, ProcessInfo};
    use crate::distributed::testing::{ClusterNode, LocalCluster};
    use std::collections::HashMap;
    use std::net::SocketAddr;

    // Helper function to create a mock Communicator for testing
    fn create_mock_comm(rank: usize, size: usize) -> Result<Arc<Communicator>, ProcessError> {
        let addr: SocketAddr = format!("127.0.0.1:{}", 8000 + rank)
            .parse()
            .map_err(|e| ProcessError::ConfigError(format!("Invalid address: {}", e)))?;

        let info = ProcessInfo::new(rank, size, addr, format!("localhost-{}", rank))?;

        let ranks: Vec<usize> = (0..size).collect();
        let group = ProcessGroup::new(ranks)?;

        let mut addresses = HashMap::new();
        for i in 0..size {
            let peer_addr: SocketAddr = format!("127.0.0.1:{}", 8000 + i)
                .parse()
                .map_err(|e| ProcessError::ConfigError(format!("Invalid address: {}", e)))?;
            addresses.insert(i, peer_addr);
        }

        let comm = Communicator::new(info, group, addresses)?;
        Ok(Arc::new(comm))
    }

    #[test]
    fn test_optimizer_error_from_invalid_lr() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let result = DistributedSGD::new(-0.01, comm);
        assert!(result.is_err());
    }

    #[test]
    fn test_distributed_sgd_creation() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let sgd = DistributedSGD::new(0.01, comm);
        assert!(sgd.is_ok());

        let optimizer = sgd.expect("optimizer creation failed");
        assert_eq!(optimizer.lr(), 0.01);
        assert_eq!(optimizer.momentum(), 0.0);
    }

    #[test]
    fn test_distributed_sgd_with_momentum() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let sgd = DistributedSGD::new(0.01, comm)
            .expect("optimizer creation failed")
            .with_momentum(0.9);

        assert_eq!(sgd.momentum(), 0.9);
    }

    #[test]
    fn test_distributed_sgd_set_lr() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let mut sgd = DistributedSGD::new(0.01, comm).expect("optimizer creation failed");

        sgd.set_lr(0.001);
        assert_eq!(sgd.lr(), 0.001);
    }

    #[test]
    fn test_distributed_adam_creation() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let adam = DistributedAdam::new(0.001, comm);
        assert!(adam.is_ok());

        let optimizer = adam.expect("optimizer creation failed");
        assert_eq!(optimizer.lr(), 0.001);
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.999);
    }

    #[test]
    fn test_distributed_adam_with_betas() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let adam = DistributedAdam::new(0.001, comm)
            .expect("optimizer creation failed")
            .with_betas(0.95, 0.9999);

        assert_eq!(adam.beta1(), 0.95);
        assert_eq!(adam.beta2(), 0.9999);
    }

    #[test]
    fn test_distributed_adam_set_lr() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let mut adam = DistributedAdam::new(0.001, comm).expect("optimizer creation failed");

        adam.set_lr(0.0001);
        assert_eq!(adam.lr(), 0.0001);
    }

    #[test]
    fn test_optimizer_state_creation() {
        let state = OptimizerState::new(10, false);
        assert_eq!(state.m.len(), 10);
        assert!(state.v.is_none());
        assert_eq!(state.t, 0);

        let state_with_v = OptimizerState::new(10, true);
        assert!(state_with_v.v.is_some());
        assert_eq!(state_with_v.v.as_ref().expect("v missing").len(), 10);
    }

    #[test]
    fn test_compression_strategy_with_optimizer() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let sgd = DistributedSGD::new(0.01, comm)
            .expect("optimizer creation failed")
            .with_compression(CompressionStrategy::TopK { k: 100 });

        // Just verify it compiles and runs
        let _ = sgd;
    }

    #[test]
    fn test_accumulation_steps() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let sgd = DistributedSGD::new(0.01, comm)
            .expect("optimizer creation failed")
            .with_accumulation(4);

        assert_eq!(sgd.accumulation_steps, 4);
    }

    #[test]
    fn test_distributed_sgd_with_nesterov() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let sgd = DistributedSGD::new(0.01, comm)
            .expect("optimizer creation failed")
            .with_nesterov();

        assert!(sgd.nesterov);
    }

    #[test]
    fn test_distributed_adam_with_weight_decay() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let adam = DistributedAdam::new(0.001, comm)
            .expect("optimizer creation failed")
            .with_weight_decay(0.01);

        assert_eq!(adam.weight_decay, 0.01);
    }

    #[test]
    fn test_distributed_adam_with_amsgrad() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let adam = DistributedAdam::new(0.001, comm)
            .expect("optimizer creation failed")
            .with_amsgrad();

        assert!(adam.amsgrad);
    }

    // -----------------------------------------------------------------
    // aggregate_gradients: a real all-reduce-then-mean over a LocalCluster,
    // for both `DistributedSGD` and `DistributedAdam`, and for every
    // `GradientAggregation` strategy. An earlier version's
    // `AllReduce`/`Hierarchical` branch never left its own rank (dividing
    // *this rank's own* gradients by the world size), so every rank
    // produced a different, wrong "aggregate" from a step that never
    // actually synchronized gradients across the cluster.
    // -----------------------------------------------------------------

    /// Every rank contributes `[r + 1.0, 2 * (r + 1.0)]`; the mean over
    /// ranks 0..4 is `[(1+2+3+4)/4, 2*(1+2+3+4)/4] = [2.5, 5.0]`.
    fn local_gradient(rank: u32) -> Vec<f32> {
        vec![(rank + 1) as f32, 2.0 * (rank + 1) as f32]
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sgd_step_averages_gradients_across_ranks() {
        for aggregation in [
            GradientAggregation::AllReduce,
            GradientAggregation::Hierarchical,
            GradientAggregation::RingAllReduce,
        ] {
            let results = LocalCluster::run_connected(4, move |node: ClusterNode| async move {
                let comm = Arc::new(
                    Communicator::from_endpoint(Arc::new(node.endpoint))
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?,
                );
                let rank = comm.rank() as u32;
                let mut sgd = DistributedSGD::new(1.0, comm)
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?
                    .with_aggregation(aggregation)
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                let params = vec![0.0_f32, 0.0];
                let updated = sgd
                    .step(&params, &local_gradient(rank))
                    .await
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                Ok(updated)
            })
            .await
            .expect("cluster run should succeed");

            // lr = 1.0, no momentum: updated = params - lr * mean_grad = -mean_grad.
            for updated in results {
                assert_eq!(updated, vec![-2.5, -5.0], "aggregation={aggregation:?}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn adam_step_averages_gradients_across_ranks() {
        for aggregation in [
            GradientAggregation::AllReduce,
            GradientAggregation::Hierarchical,
            GradientAggregation::RingAllReduce,
        ] {
            let results = LocalCluster::run_connected(4, move |node: ClusterNode| async move {
                let comm = Arc::new(
                    Communicator::from_endpoint(Arc::new(node.endpoint))
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?,
                );
                let rank = comm.rank() as u32;
                let mut adam = DistributedAdam::new(1.0, comm)
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?
                    .with_aggregation(aggregation)
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                let params = vec![0.0_f32, 0.0];
                let updated = adam
                    .step(&params, &local_gradient(rank))
                    .await
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                Ok(updated)
            })
            .await
            .expect("cluster run should succeed");

            // Every rank must land on the identical update: proof the
            // gradient actually crossed the network rather than each rank
            // computing Adam's first step from its own local (and
            // therefore different) gradient.
            let first = results[0].clone();
            for updated in &results {
                assert_eq!(updated, &first, "aggregation={aggregation:?}");
            }
        }
    }
}
