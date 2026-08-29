//! Data Parallelism for Distributed Training
//!
//! This module implements data-parallel distributed training patterns where the model
//! is replicated across workers and data is partitioned.
//!
//! # Features
//!
//! - **Distributed Data Loader**: Automatic data sharding across workers
//! - **Mini-Batch Distribution**: Efficient batch partitioning
//! - **Gradient Aggregation**: AllReduce and Ring-AllReduce strategies
//! - **Synchronous SGD**: Lock-step gradient updates, real multi-rank
//!   all-reduce via [`SyncDataParallel`]
//! - **Asynchronous SGD**: Non-blocking parameter updates via
//!   [`AsyncDataParallel`] — single-process only, see its docs
//! - **Gradient Accumulation**: Multi-step gradient aggregation
//!
//! # Data Parallel Patterns
//!
//! ## Synchronous Data Parallel
//!
//! Real, multi-rank: every rank in `Step 3` genuinely exchanges gradients
//! with every other rank over the network (via
//! [`SyncDataParallel::aggregate_gradients`], which all-reduces then
//! divides by the world size — every rank ends this step holding the same
//! mean gradient).
//! ```text
//! Step 1: Forward pass on local data
//! Step 2: Compute gradients
//! Step 3: AllReduce gradients
//! Step 4: Update parameters synchronously
//! ```
//!
//! ## Asynchronous Data Parallel
//!
//! [`AsyncDataParallel`] models this shape, but — see its docs and
//! [`ParameterServer`]'s — "Worker 0"/"Worker 1"/"Parameter Server" below
//! are logical roles sharing one process today, not separate ranks: nothing
//! here yet sends a push or a pull across a real network boundary.
//! ```text
//! Worker 0: compute → push gradients → continue
//! Worker 1: compute → push gradients → continue
//! Parameter Server: aggregate and update
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::data_parallel::*;
//! use numrs2::distributed::process::*;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), DataParallelError> {
//! let world = init().await?;
//!
//! // Create distributed data loader
//! let dataset = vec![1.0; 10000];
//! let mut loader = DistributedDataLoader::new(
//!     dataset,
//!     32,  // batch size
//!     &world,
//!     ShardingStrategy::Block
//! )?;
//!
//! // Synchronous training step
//! let sync_trainer = SyncDataParallel::new(Arc::new(world), GradientAggregation::AllReduce)?;
//!
//! for batch in loader.iter() {
//!     // Forward pass and compute gradients
//!     let gradients = vec![0.1; 1000];
//!
//!     // Aggregate and update
//!     let averaged = sync_trainer.aggregate_gradients(&gradients).await?;
//!     // Apply updates...
//! }
//! # Ok(())
//! # }
//! ```

use super::collective::{allreduce, ReduceOp};
use super::communication::CommunicationError;
use super::coordinator::{CoordinatorError, ParameterServer, RingAllReduce};
use super::process::{Communicator, ProcessError};
use crate::error::NumRs2Error;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;

/// Errors in data parallel operations
#[derive(Error, Debug)]
pub enum DataParallelError {
    #[error("Process error: {0}")]
    Process(#[from] ProcessError),

    #[error("Communication error: {0}")]
    Communication(#[from] CommunicationError),

    #[error("Coordinator error: {0}")]
    Coordinator(#[from] CoordinatorError),

    #[error("Invalid batch size: {0}")]
    InvalidBatchSize(usize),

    #[error("Dataset size mismatch: expected {expected}, got {actual}")]
    DatasetSizeMismatch { expected: usize, actual: usize },

    #[error("Gradient aggregation error: {0}")]
    AggregationError(String),

    #[error("Sharding error: {0}")]
    ShardingError(String),
}

impl From<DataParallelError> for NumRs2Error {
    fn from(err: DataParallelError) -> Self {
        NumRs2Error::DistributedComputing(err.to_string())
    }
}

/// Strategy for sharding data across workers
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, oxicode::Encode, oxicode::Decode,
)]
pub enum ShardingStrategy {
    /// Block sharding: contiguous chunks
    Block,

    /// Cyclic sharding: round-robin distribution
    Cyclic,

    /// Random sharding: random assignment
    Random,
}

/// Strategy for gradient aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientAggregation {
    /// Standard AllReduce (tree-based)
    AllReduce,

    /// Ring-based AllReduce (bandwidth-optimal)
    RingAllReduce,

    /// Hierarchical reduce-broadcast
    Hierarchical,
}

/// Distributed data loader with automatic sharding
pub struct DistributedDataLoader<T> {
    /// Local shard of data
    local_data: Vec<T>,

    /// Batch size
    batch_size: usize,

    /// Current position in local data
    position: usize,

    /// Global dataset size
    global_size: usize,

    /// Rank in communicator
    rank: usize,

    /// Total number of workers
    world_size: usize,

    /// Sharding strategy
    strategy: ShardingStrategy,
}

impl<T: Clone> DistributedDataLoader<T> {
    /// Create new distributed data loader
    pub fn new(
        global_data: Vec<T>,
        batch_size: usize,
        communicator: &Communicator,
        strategy: ShardingStrategy,
    ) -> Result<Self, DataParallelError> {
        if batch_size == 0 {
            return Err(DataParallelError::InvalidBatchSize(batch_size));
        }

        let rank = communicator.rank();
        let world_size = communicator.size();
        let global_size = global_data.len();

        // Shard data according to strategy
        let local_data = Self::shard_data(global_data, rank, world_size, strategy)?;

        Ok(Self {
            local_data,
            batch_size,
            position: 0,
            global_size,
            rank,
            world_size,
            strategy,
        })
    }

    /// Shard data according to strategy
    fn shard_data(
        mut data: Vec<T>,
        rank: usize,
        world_size: usize,
        strategy: ShardingStrategy,
    ) -> Result<Vec<T>, DataParallelError> {
        match strategy {
            ShardingStrategy::Block => {
                let chunk_size = data.len().div_ceil(world_size);
                let start = rank * chunk_size;
                let end = (start + chunk_size).min(data.len());

                if start >= data.len() {
                    Ok(Vec::new())
                } else {
                    Ok(data.drain(start..end).collect())
                }
            }

            ShardingStrategy::Cyclic => {
                let local: Vec<T> = data
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % world_size == rank)
                    .map(|(_, item)| item.clone())
                    .collect();
                Ok(local)
            }

            ShardingStrategy::Random => {
                // For reproducibility, use deterministic "random" sharding
                // based on index hash
                let local: Vec<T> = data
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        let hash = i.wrapping_mul(2654435761) % world_size;
                        hash == rank
                    })
                    .map(|(_, item)| item.clone())
                    .collect();
                Ok(local)
            }
        }
    }

    /// Get next batch
    pub fn next_batch(&mut self) -> Option<Vec<T>> {
        if self.position >= self.local_data.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.local_data.len());
        let batch = self.local_data[self.position..end].to_vec();
        self.position = end;

        Some(batch)
    }

    /// Reset loader to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Get number of local batches
    pub fn num_batches(&self) -> usize {
        self.local_data.len().div_ceil(self.batch_size)
    }

    /// Get local data size
    pub fn local_size(&self) -> usize {
        self.local_data.len()
    }

    /// Get global data size
    pub fn global_size(&self) -> usize {
        self.global_size
    }

    /// Rank of this loader's shard within the communicator it was built from
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Total number of workers this loader's data was sharded across
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Sharding strategy this loader was constructed with
    pub fn strategy(&self) -> ShardingStrategy {
        self.strategy
    }

    /// Create iterator over batches
    pub fn iter(&mut self) -> DataLoaderIterator<'_, T> {
        DataLoaderIterator { loader: self }
    }
}

/// Iterator over data loader batches
pub struct DataLoaderIterator<'a, T> {
    loader: &'a mut DistributedDataLoader<T>,
}

impl<'a, T: Clone> Iterator for DataLoaderIterator<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.loader.next_batch()
    }
}

/// Synchronous data parallel trainer
pub struct SyncDataParallel {
    /// Communicator
    communicator: Arc<Communicator>,

    /// Gradient aggregation strategy
    aggregation: GradientAggregation,

    /// Ring-allreduce coordinator (if using ring strategy)
    ring_reducer: Option<RingAllReduce>,

    /// Gradient accumulation steps
    accumulation_steps: usize,

    /// Current accumulation count
    current_step: Arc<Mutex<usize>>,

    /// Accumulated gradients
    gradient_buffer: Arc<Mutex<Vec<f32>>>,
}

impl SyncDataParallel {
    /// Create new synchronous data parallel trainer
    pub fn new(
        communicator: Arc<Communicator>,
        aggregation: GradientAggregation,
    ) -> Result<Self, DataParallelError> {
        let ring_reducer = if aggregation == GradientAggregation::RingAllReduce {
            Some(RingAllReduce::new(communicator.clone())?)
        } else {
            None
        };

        Ok(Self {
            communicator,
            aggregation,
            ring_reducer,
            accumulation_steps: 1,
            current_step: Arc::new(Mutex::new(0)),
            gradient_buffer: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Set gradient accumulation steps
    pub fn with_accumulation(mut self, steps: usize) -> Self {
        self.accumulation_steps = steps;
        self
    }

    /// Aggregate gradients across all workers
    pub async fn aggregate_gradients(
        &self,
        local_gradients: &[f32],
    ) -> Result<Vec<f32>, DataParallelError> {
        // Accumulate gradients if multi-step
        if self.accumulation_steps > 1 {
            let mut buffer = self.gradient_buffer.lock().await;
            if buffer.is_empty() {
                buffer.resize(local_gradients.len(), 0.0);
            }

            for (acc, &grad) in buffer.iter_mut().zip(local_gradients.iter()) {
                *acc += grad;
            }

            let mut step = self.current_step.lock().await;
            *step += 1;

            if *step < self.accumulation_steps {
                // Not ready to aggregate yet
                return Ok(vec![0.0; local_gradients.len()]);
            }

            // Reset for next accumulation
            *step = 0;
            let accumulated = buffer.clone();
            buffer.fill(0.0);
            drop(buffer);
            drop(step);

            // Continue with accumulated gradients
            self.aggregate_impl(&accumulated).await
        } else {
            self.aggregate_impl(local_gradients).await
        }
    }

    /// Internal gradient aggregation implementation.
    ///
    /// A real all-reduce **sum** across every rank, then divided by the
    /// world size — every rank contributes and every rank gets the same
    /// mean gradient, which is what every doc example calling the result
    /// `averaged` expects. An earlier version's `AllReduce`/`Hierarchical`
    /// branch never talked to another rank at all: it divided *this rank's
    /// own* gradients by the world size and returned that, so every rank
    /// silently produced a different, wrong result (that of `RingAllReduce`
    /// was wrong the other way — a real cross-rank sum, but never divided
    /// down to a mean).
    async fn aggregate_impl(&self, gradients: &[f32]) -> Result<Vec<f32>, DataParallelError> {
        let world_size = self.communicator.size() as f32;
        let summed = match self.aggregation {
            GradientAggregation::RingAllReduce => {
                let reducer = self.ring_reducer.as_ref().ok_or_else(|| {
                    DataParallelError::AggregationError("Ring reducer not initialized".to_string())
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

    /// Get current accumulation step
    pub async fn current_accumulation_step(&self) -> usize {
        *self.current_step.lock().await
    }

    /// Get total accumulation steps
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }
}

/// Asynchronous data parallel trainer.
///
/// Thin wrapper over [`ParameterServer`]; see that type's docs for why
/// [`Self::new`] rejects a multi-rank `communicator`.
pub struct AsyncDataParallel {
    /// Parameter server
    ps: ParameterServer,

    /// Staleness threshold for stale gradient rejection
    staleness_threshold: Option<u64>,
}

impl AsyncDataParallel {
    /// Create new asynchronous data parallel trainer.
    ///
    /// # Errors
    ///
    /// See [`ParameterServer::new`]: this rejects the same multi-rank
    /// `communicator`, for the same reason.
    pub fn new(communicator: Arc<Communicator>, num_ps: usize) -> Result<Self, DataParallelError> {
        let ps = ParameterServer::new(communicator, num_ps)?;

        Ok(Self {
            ps,
            staleness_threshold: None,
        })
    }

    /// Set staleness threshold
    pub fn with_staleness_threshold(mut self, threshold: u64) -> Self {
        self.staleness_threshold = Some(threshold);
        self
    }

    /// Push gradients asynchronously
    pub async fn push_gradients(
        &self,
        parameter_key: &str,
        gradients: &[f32],
    ) -> Result<(), DataParallelError> {
        self.ps.push_gradients(parameter_key, gradients).await?;
        Ok(())
    }

    /// Pull parameters asynchronously
    pub async fn pull_parameters(
        &self,
        parameter_key: &str,
    ) -> Result<Vec<f32>, DataParallelError> {
        Ok(self.ps.pull_parameters(parameter_key).await?)
    }

    /// Check if gradients are too stale
    pub async fn is_stale(
        &self,
        parameter_key: &str,
        local_version: u64,
    ) -> Result<bool, DataParallelError> {
        if let Some(threshold) = self.staleness_threshold {
            let current_version = self.ps.get_version(parameter_key).await?;
            Ok(current_version > local_version + threshold)
        } else {
            Ok(false)
        }
    }

    /// Apply accumulated gradients
    pub async fn apply_gradients(
        &self,
        parameter_key: &str,
        learning_rate: f32,
    ) -> Result<(), DataParallelError> {
        self.ps
            .apply_gradients(parameter_key, learning_rate)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::process::{ProcessGroup, ProcessInfo};
    use std::collections::HashMap;
    use std::net::SocketAddr;

    // Helper function to create a mock Communicator for testing
    fn create_mock_comm(rank: usize, size: usize) -> Result<Communicator, ProcessError> {
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

        Communicator::new(info, group, addresses)
    }

    #[test]
    fn test_sharding_strategy_serialization() {
        let strategies = vec![
            ShardingStrategy::Block,
            ShardingStrategy::Cyclic,
            ShardingStrategy::Random,
        ];

        for strategy in strategies {
            let serialized = oxicode::encode_to_vec(&strategy);
            assert!(serialized.is_ok());

            let bytes = serialized.expect("serialization failed");
            let result = oxicode::decode_from_slice::<ShardingStrategy>(&bytes);
            assert!(result.is_ok());
            let (deserialized, _) = result.expect("deserialization failed");
            assert_eq!(
                std::mem::discriminant(&strategy),
                std::mem::discriminant(&deserialized)
            );
        }
    }

    #[test]
    fn test_gradient_aggregation_eq() {
        assert_eq!(
            GradientAggregation::AllReduce,
            GradientAggregation::AllReduce
        );
        assert_ne!(
            GradientAggregation::AllReduce,
            GradientAggregation::RingAllReduce
        );
    }

    #[test]
    fn test_data_loader_block_sharding() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Simulate rank 0 of 4 workers
        let shard =
            DistributedDataLoader::<f32>::shard_data(data.clone(), 0, 4, ShardingStrategy::Block);
        assert!(shard.is_ok());

        let local = shard.expect("sharding failed");
        assert_eq!(local.len(), 25);
        assert_eq!(local[0], 0.0);
        assert_eq!(local[24], 24.0);
    }

    #[test]
    fn test_data_loader_cyclic_sharding() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let shard =
            DistributedDataLoader::<f32>::shard_data(data.clone(), 0, 4, ShardingStrategy::Cyclic);
        assert!(shard.is_ok());

        let local = shard.expect("sharding failed");
        assert_eq!(local.len(), 25);
        assert_eq!(local[0], 0.0);
        assert_eq!(local[1], 4.0);
        assert_eq!(local[2], 8.0);
    }

    #[test]
    fn test_data_loader_random_sharding() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let shard =
            DistributedDataLoader::<f32>::shard_data(data.clone(), 0, 4, ShardingStrategy::Random);
        assert!(shard.is_ok());

        let local = shard.expect("sharding failed");
        assert!(!local.is_empty());
        assert!(local.len() <= 100);
    }

    /// [`AsyncDataParallel::new`] must propagate
    /// [`ParameterServer::new`]'s multi-rank rejection rather than
    /// swallowing it — see that type's docs on why a multi-rank
    /// `ParameterServer` would silently desynchronize instead of erroring.
    #[test]
    fn async_data_parallel_rejects_a_multi_rank_communicator() {
        let comm = create_mock_comm(0, 4).expect("Failed to create mock communicator");
        // `expect_err` needs `AsyncDataParallel: Debug` (the success case),
        // which this type deliberately does not derive — match instead.
        match AsyncDataParallel::new(Arc::new(comm), 2) {
            Err(DataParallelError::Coordinator(CoordinatorError::Configuration(_))) => {}
            Err(other) => {
                panic!("expected DataParallelError::Coordinator(Configuration), got {other}")
            }
            Ok(_) => panic!("a 4-rank communicator must be rejected"),
        }
    }

    #[test]
    fn async_data_parallel_accepts_a_single_rank_communicator() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        assert!(AsyncDataParallel::new(Arc::new(comm), 1).is_ok());
    }

    #[test]
    fn test_invalid_batch_size() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let data = vec![1.0; 100];
        let result = DistributedDataLoader::new(data, 0, &comm, ShardingStrategy::Block);

        assert!(result.is_err());
        match result.err() {
            Some(DataParallelError::InvalidBatchSize(0)) => (),
            _ => panic!("Expected InvalidBatchSize error"),
        }
    }

    #[test]
    fn test_data_loader_num_batches() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let data = vec![1.0; 100];
        let loader = DistributedDataLoader::new(data, 32, &comm, ShardingStrategy::Block);

        assert!(loader.is_ok());
        let dl = loader.expect("loader creation failed");

        // 100 samples / 32 batch size = 4 batches (rounded up)
        assert_eq!(dl.num_batches(), 4);
    }

    #[test]
    fn test_data_loader_sizes() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let data = vec![1.0; 100];
        let loader = DistributedDataLoader::new(data, 32, &comm, ShardingStrategy::Block);

        assert!(loader.is_ok());
        let dl = loader.expect("loader creation failed");

        assert_eq!(dl.local_size(), 100);
        assert_eq!(dl.global_size(), 100);
    }

    #[test]
    fn test_data_loader_next_batch() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut loader = DistributedDataLoader::new(data, 2, &comm, ShardingStrategy::Block)
            .expect("loader creation failed");

        let batch1 = loader.next_batch();
        assert!(batch1.is_some());
        assert_eq!(batch1.expect("batch1").len(), 2);

        let batch2 = loader.next_batch();
        assert!(batch2.is_some());
        assert_eq!(batch2.expect("batch2").len(), 2);

        let batch3 = loader.next_batch();
        assert!(batch3.is_some());
        assert_eq!(batch3.expect("batch3").len(), 1);

        let batch4 = loader.next_batch();
        assert!(batch4.is_none());
    }

    #[test]
    fn test_data_loader_reset() {
        let comm = create_mock_comm(0, 1).expect("Failed to create mock communicator");
        let data = vec![1.0, 2.0, 3.0];
        let mut loader = DistributedDataLoader::new(data, 2, &comm, ShardingStrategy::Block)
            .expect("loader creation failed");

        let _ = loader.next_batch();
        let _ = loader.next_batch();
        assert!(loader.next_batch().is_none());

        loader.reset();
        let batch = loader.next_batch();
        assert!(batch.is_some());
    }

    // -----------------------------------------------------------------
    // aggregate_impl: a real all-reduce-then-mean over a LocalCluster, for
    // every `GradientAggregation` strategy. An earlier version's
    // `AllReduce`/`Hierarchical` branch never left its own rank (dividing
    // *this rank's own* gradients by the world size), so every rank
    // produced a different, wrong result; `RingAllReduce` really summed
    // across ranks but never divided down to a mean. Both are wrong in
    // exactly the way a genuine cross-rank average would catch.
    // -----------------------------------------------------------------

    use crate::distributed::testing::{ClusterNode, LocalCluster};

    async fn averages_four_ranks_gradients(
        aggregation: GradientAggregation,
    ) -> Vec<Result<Vec<f32>, DataParallelError>> {
        LocalCluster::run_connected(4, move |node: ClusterNode| async move {
            let comm = Arc::new(Communicator::from_endpoint(Arc::new(node.endpoint))?);
            let rank = comm.rank();
            let trainer = SyncDataParallel::new(comm, aggregation)
                .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
            // Rank r contributes [r + 1.0, 2 * (r + 1.0)]; the mean over
            // ranks 0..4 is [(1+2+3+4)/4, 2*(1+2+3+4)/4] = [2.5, 5.0].
            let local = vec![(rank + 1) as f32, 2.0 * (rank + 1) as f32];
            Ok(trainer.aggregate_gradients(&local).await)
        })
        .await
        .expect("cluster run should succeed")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregate_gradients_averages_across_ranks_with_all_reduce() {
        for result in averages_four_ranks_gradients(GradientAggregation::AllReduce).await {
            let averaged = result.expect("aggregate_gradients should succeed");
            assert_eq!(averaged, vec![2.5, 5.0]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregate_gradients_averages_across_ranks_with_hierarchical() {
        for result in averages_four_ranks_gradients(GradientAggregation::Hierarchical).await {
            let averaged = result.expect("aggregate_gradients should succeed");
            assert_eq!(averaged, vec![2.5, 5.0]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregate_gradients_averages_across_ranks_with_ring_all_reduce() {
        for result in averages_four_ranks_gradients(GradientAggregation::RingAllReduce).await {
            let averaged = result.expect("aggregate_gradients should succeed");
            assert_eq!(averaged, vec![2.5, 5.0]);
        }
    }
}
