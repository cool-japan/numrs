//! Model Parallelism for Large-Scale Deep Learning
//!
//! This module implements model-parallel training patterns where the model is
//! partitioned across workers, enabling training of models too large for a single device.
//!
//! # Features
//!
//! - **Layer-wise Partitioning**: Split model by layers
//! - **Pipeline Parallelism**: GPipe-style micro-batching
//! - **Tensor Parallelism**: Megatron-style intra-layer partitioning
//! - **Activation Checkpointing**: Memory-efficient backpropagation
//! - **Gradient Accumulation**: Multi-microbatch training
//!
//! # Parallelism Patterns
//!
//! ## Pipeline Parallelism (GPipe)
//! ```text
//! Time →
//! GPU0: [F0] [F1] [F2] [B0] [B1] [B2]
//! GPU1:      [F0] [F1] [F2] [B0] [B1] [B2]
//! GPU2:           [F0] [F1] [F2] [B0] [B1]
//! (F=Forward, B=Backward, numbers=microbatch)
//! ```
//!
//! ## Tensor Parallelism (Megatron)
//! ```text
//! Input → [Split] → GPU0: Linear_A
//!                  GPU1: Linear_B
//!        [Concat] → Output
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::model_parallel::*;
//! use numrs2::distributed::process::*;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), ModelParallelError> {
//! let world = init().await?;
//!
//! // Pipeline parallelism - split model into 4 stages
//! let pipeline = PipelineParallel::new(
//!     Arc::new(world.clone()),
//!     4,  // number of pipeline stages
//!     8,  // number of microbatches
//! )?;
//!
//! // Tensor parallelism - partition large layers
//! let tensor_parallel = TensorParallel::new(
//!     Arc::new(world),
//!     PartitionStrategy::ColumnWise,
//! )?;
//!
//! // Activation checkpointing for memory efficiency
//! let checkpointer = ActivationCheckpointer::new(2)?; // checkpoint every 2 layers
//! # Ok(())
//! # }
//! ```

use super::communication::{
    AsyncCommunicator, CommunicationError, CompressionStrategy, MessagePriority, TensorMessage,
};
use super::coordinator::CoordinatorError;
use super::process::{Communicator, ProcessError};
use crate::error::NumRs2Error;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};

/// Errors in model parallel operations
#[derive(Error, Debug)]
pub enum ModelParallelError {
    #[error("Process error: {0}")]
    Process(#[from] ProcessError),

    #[error("Communication error: {0}")]
    Communication(#[from] CommunicationError),

    #[error("Coordinator error: {0}")]
    Coordinator(#[from] CoordinatorError),

    #[error("Invalid stage assignment: stage {stage} out of {total}")]
    InvalidStage { stage: usize, total: usize },

    #[error("Partition error: {0}")]
    PartitionError(String),

    #[error("Pipeline error: {0}")]
    PipelineError(String),

    #[error("Checkpoint error: {0}")]
    CheckpointError(String),

    #[error("Invalid microbatch: {0}")]
    InvalidMicrobatch(String),
}

impl From<ModelParallelError> for NumRs2Error {
    fn from(err: ModelParallelError) -> Self {
        NumRs2Error::DistributedComputing(err.to_string())
    }
}

/// Tensor partitioning strategy
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, oxicode::Encode, oxicode::Decode,
)]
pub enum PartitionStrategy {
    /// Split along columns (for linear layers)
    ColumnWise,

    /// Split along rows (for linear layers)
    RowWise,

    /// Split along batch dimension
    BatchWise,

    /// Split along sequence dimension (for transformers)
    SequenceWise,
}

/// Pipeline stage information
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage ID
    pub stage_id: usize,

    /// Total number of stages
    pub num_stages: usize,

    /// Ranks assigned to this stage
    pub ranks: Vec<usize>,

    /// Previous stage (None for first stage)
    pub prev_stage: Option<usize>,

    /// Next stage (None for last stage)
    pub next_stage: Option<usize>,
}

impl PipelineStage {
    /// Create new pipeline stage
    pub fn new(stage_id: usize, num_stages: usize, ranks: Vec<usize>) -> Self {
        let prev_stage = if stage_id > 0 {
            Some(stage_id - 1)
        } else {
            None
        };

        let next_stage = if stage_id < num_stages - 1 {
            Some(stage_id + 1)
        } else {
            None
        };

        Self {
            stage_id,
            num_stages,
            ranks,
            prev_stage,
            next_stage,
        }
    }

    /// Check if this is the first stage
    pub fn is_first(&self) -> bool {
        self.stage_id == 0
    }

    /// Check if this is the last stage
    pub fn is_last(&self) -> bool {
        self.stage_id == self.num_stages - 1
    }
}

/// Microbatch for pipeline parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Microbatch<T> {
    /// Microbatch ID
    pub id: usize,

    /// Data
    pub data: Vec<T>,

    /// Shape
    pub shape: Vec<usize>,

    /// Stage where this microbatch is currently
    pub current_stage: usize,
}

impl<T: Clone> Microbatch<T> {
    /// Create new microbatch
    pub fn new(id: usize, data: Vec<T>, shape: Vec<usize>) -> Self {
        Self {
            id,
            data,
            shape,
            current_stage: 0,
        }
    }

    /// Move to next stage
    pub fn advance_stage(&mut self) {
        self.current_stage += 1;
    }
}

/// Pipeline parallel coordinator
pub struct PipelineParallel {
    /// Communicator
    communicator: Arc<Communicator>,

    /// Async communicator
    async_comm: AsyncCommunicator,

    /// This worker's pipeline stage
    stage: PipelineStage,

    /// Number of microbatches
    num_microbatches: usize,

    /// Forward activation buffer
    forward_buffer: Arc<Mutex<HashMap<usize, Vec<f32>>>>,

    /// Backward gradient buffer
    backward_buffer: Arc<Mutex<HashMap<usize, Vec<f32>>>>,

    /// Pipeline schedule
    schedule: PipelineSchedule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineSchedule {
    /// GPipe: forward all microbatches, then backward all
    GPipe,

    /// PipeDream: interleaved 1F1B (one forward, one backward)
    OneFOneB,
}

impl PipelineParallel {
    /// Create new pipeline parallel coordinator
    pub fn new(
        communicator: Arc<Communicator>,
        num_stages: usize,
        num_microbatches: usize,
    ) -> Result<Self, ModelParallelError> {
        let async_comm = AsyncCommunicator::new(communicator.clone())?;

        let rank = communicator.rank();
        let world_size = communicator.size();

        // Assign ranks to stages
        let ranks_per_stage = world_size.div_ceil(num_stages);
        let stage_id = rank / ranks_per_stage;

        if stage_id >= num_stages {
            return Err(ModelParallelError::InvalidStage {
                stage: stage_id,
                total: num_stages,
            });
        }

        // Calculate ranks in this stage
        let stage_start = stage_id * ranks_per_stage;
        let stage_end = (stage_start + ranks_per_stage).min(world_size);
        let ranks: Vec<usize> = (stage_start..stage_end).collect();

        let stage = PipelineStage::new(stage_id, num_stages, ranks);

        Ok(Self {
            communicator,
            async_comm,
            stage,
            num_microbatches,
            forward_buffer: Arc::new(Mutex::new(HashMap::new())),
            backward_buffer: Arc::new(Mutex::new(HashMap::new())),
            schedule: PipelineSchedule::GPipe,
        })
    }

    /// Send forward activations to next stage
    pub async fn send_forward(
        &self,
        microbatch_id: usize,
        activations: &[f32],
    ) -> Result<(), ModelParallelError> {
        if let Some(next_stage) = self.stage.next_stage {
            let next_rank = next_stage * self.communicator.size().div_ceil(self.stage.num_stages);

            let msg = TensorMessage::new(
                activations.to_vec(),
                CompressionStrategy::None,
                MessagePriority::High,
            )
            .with_tag(microbatch_id as u32);

            self.async_comm.isend(msg, next_rank).await?;
        }

        Ok(())
    }

    /// Receive forward activations from previous stage
    pub async fn recv_forward(&self, microbatch_id: usize) -> Result<Vec<f32>, ModelParallelError> {
        if let Some(prev_stage) = self.stage.prev_stage {
            let prev_rank = prev_stage * self.communicator.size().div_ceil(self.stage.num_stages);

            // Check buffer first
            let mut buffer = self.forward_buffer.lock().await;
            if let Some(data) = buffer.remove(&microbatch_id) {
                return Ok(data);
            }
            drop(buffer);

            // Would receive from previous stage in real implementation
            let _ = prev_rank;
            Ok(vec![0.0; 10]) // Placeholder
        } else {
            Err(ModelParallelError::PipelineError(
                "No previous stage to receive from".to_string(),
            ))
        }
    }

    /// Send backward gradients to previous stage
    pub async fn send_backward(
        &self,
        microbatch_id: usize,
        gradients: &[f32],
    ) -> Result<(), ModelParallelError> {
        if let Some(prev_stage) = self.stage.prev_stage {
            let prev_rank = prev_stage * self.communicator.size().div_ceil(self.stage.num_stages);

            let msg = TensorMessage::new(
                gradients.to_vec(),
                CompressionStrategy::None,
                MessagePriority::High,
            )
            .with_tag(microbatch_id as u32);

            self.async_comm.isend(msg, prev_rank).await?;
        }

        Ok(())
    }

    /// Receive backward gradients from next stage
    pub async fn recv_backward(
        &self,
        microbatch_id: usize,
    ) -> Result<Vec<f32>, ModelParallelError> {
        if let Some(next_stage) = self.stage.next_stage {
            let next_rank = next_stage * self.communicator.size().div_ceil(self.stage.num_stages);

            // Check buffer first
            let mut buffer = self.backward_buffer.lock().await;
            if let Some(data) = buffer.remove(&microbatch_id) {
                return Ok(data);
            }
            drop(buffer);

            // Would receive from next stage in real implementation
            let _ = next_rank;
            Ok(vec![0.0; 10]) // Placeholder
        } else {
            Err(ModelParallelError::PipelineError(
                "No next stage to receive from".to_string(),
            ))
        }
    }

    /// Get stage information
    pub fn stage(&self) -> &PipelineStage {
        &self.stage
    }

    /// Get number of microbatches
    pub fn num_microbatches(&self) -> usize {
        self.num_microbatches
    }
}

/// Tensor parallel coordinator
pub struct TensorParallel {
    /// Communicator
    communicator: Arc<Communicator>,

    /// Async communicator
    async_comm: AsyncCommunicator,

    /// Partition strategy
    strategy: PartitionStrategy,

    /// Tensor parallel group size
    tp_size: usize,

    /// Rank within tensor parallel group
    tp_rank: usize,
}

impl TensorParallel {
    /// Create new tensor parallel coordinator
    pub fn new(
        communicator: Arc<Communicator>,
        strategy: PartitionStrategy,
    ) -> Result<Self, ModelParallelError> {
        let async_comm = AsyncCommunicator::new(communicator.clone())?;
        let tp_size = communicator.size();
        let tp_rank = communicator.rank();

        Ok(Self {
            communicator,
            async_comm,
            strategy,
            tp_size,
            tp_rank,
        })
    }

    /// Partition tensor according to strategy
    pub fn partition(
        &self,
        tensor: &[f32],
        shape: &[usize],
    ) -> Result<Vec<f32>, ModelParallelError> {
        match self.strategy {
            PartitionStrategy::ColumnWise => {
                if shape.len() != 2 {
                    return Err(ModelParallelError::PartitionError(
                        "ColumnWise partition requires 2D tensor".to_string(),
                    ));
                }

                let cols = shape[1];
                let cols_per_rank = cols.div_ceil(self.tp_size);
                let start_col = self.tp_rank * cols_per_rank;
                let end_col = (start_col + cols_per_rank).min(cols);

                // Extract columns for this rank
                let mut partition = Vec::new();
                for row in 0..shape[0] {
                    for col in start_col..end_col {
                        let idx = row * cols + col;
                        if idx < tensor.len() {
                            partition.push(tensor[idx]);
                        }
                    }
                }

                Ok(partition)
            }

            PartitionStrategy::RowWise => {
                if shape.len() != 2 {
                    return Err(ModelParallelError::PartitionError(
                        "RowWise partition requires 2D tensor".to_string(),
                    ));
                }

                let rows = shape[0];
                let rows_per_rank = rows.div_ceil(self.tp_size);
                let start_row = self.tp_rank * rows_per_rank;
                let end_row = (start_row + rows_per_rank).min(rows);

                let cols = shape[1];
                let mut partition = Vec::new();

                for row in start_row..end_row {
                    for col in 0..cols {
                        let idx = row * cols + col;
                        if idx < tensor.len() {
                            partition.push(tensor[idx]);
                        }
                    }
                }

                Ok(partition)
            }

            PartitionStrategy::BatchWise | PartitionStrategy::SequenceWise => {
                // Simple block partitioning
                let chunk_size = tensor.len().div_ceil(self.tp_size);
                let start = self.tp_rank * chunk_size;
                let end = (start + chunk_size).min(tensor.len());

                Ok(tensor[start..end].to_vec())
            }
        }
    }

    /// All-gather partitioned tensors
    pub async fn gather(&self, local_tensor: &[f32]) -> Result<Vec<f32>, ModelParallelError> {
        // Would perform actual all-gather in real implementation
        // For now, just return local tensor
        Ok(local_tensor.to_vec())
    }

    /// Get partition strategy
    pub fn strategy(&self) -> PartitionStrategy {
        self.strategy
    }

    /// Get tensor parallel size
    pub fn tp_size(&self) -> usize {
        self.tp_size
    }

    /// Get tensor parallel rank
    pub fn tp_rank(&self) -> usize {
        self.tp_rank
    }
}

/// Activation checkpointer for memory-efficient training
pub struct ActivationCheckpointer {
    /// Checkpoint interval (checkpoint every N layers)
    interval: usize,

    /// Stored checkpoints (layer_id -> activations)
    checkpoints: Arc<RwLock<HashMap<usize, Vec<f32>>>>,

    /// Recomputation count
    recomputation_count: Arc<Mutex<usize>>,
}

impl ActivationCheckpointer {
    /// Create new activation checkpointer
    pub fn new(interval: usize) -> Result<Self, ModelParallelError> {
        Ok(Self {
            interval,
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            recomputation_count: Arc::new(Mutex::new(0)),
        })
    }

    /// Check if layer should be checkpointed
    pub fn should_checkpoint(&self, layer_id: usize) -> bool {
        layer_id.is_multiple_of(self.interval)
    }

    /// Store checkpoint for layer
    pub async fn checkpoint(&self, layer_id: usize, activations: Vec<f32>) {
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.insert(layer_id, activations);
    }

    /// Retrieve checkpoint
    pub async fn get_checkpoint(&self, layer_id: usize) -> Option<Vec<f32>> {
        let checkpoints = self.checkpoints.read().await;
        checkpoints.get(&layer_id).cloned()
    }

    /// Clear all checkpoints
    pub async fn clear(&self) {
        let mut checkpoints = self.checkpoints.write().await;
        checkpoints.clear();
    }

    /// Get recomputation count
    pub async fn recomputation_count(&self) -> usize {
        *self.recomputation_count.lock().await
    }

    /// Increment recomputation count
    pub async fn increment_recomputation(&self) {
        let mut count = self.recomputation_count.lock().await;
        *count += 1;
    }

    /// Get checkpoint interval
    pub fn interval(&self) -> usize {
        self.interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_strategy_serialization() {
        let strategies = vec![
            PartitionStrategy::ColumnWise,
            PartitionStrategy::RowWise,
            PartitionStrategy::BatchWise,
            PartitionStrategy::SequenceWise,
        ];

        for strategy in strategies {
            let serialized = oxicode::encode_to_vec(&strategy);
            assert!(serialized.is_ok());

            let bytes = serialized.expect("serialization failed");
            let result = oxicode::decode_from_slice::<PartitionStrategy>(&bytes);
            assert!(result.is_ok());
            let (deserialized, _) = result.expect("deserialization failed");
            assert_eq!(
                std::mem::discriminant(&strategy),
                std::mem::discriminant(&deserialized)
            );
        }
    }

    #[test]
    fn test_pipeline_stage_creation() {
        let stage = PipelineStage::new(1, 4, vec![2, 3]);

        assert_eq!(stage.stage_id, 1);
        assert_eq!(stage.num_stages, 4);
        assert_eq!(stage.ranks, vec![2, 3]);
        assert_eq!(stage.prev_stage, Some(0));
        assert_eq!(stage.next_stage, Some(2));
        assert!(!stage.is_first());
        assert!(!stage.is_last());
    }

    #[test]
    fn test_pipeline_stage_first() {
        let stage = PipelineStage::new(0, 4, vec![0]);

        assert!(stage.is_first());
        assert!(!stage.is_last());
        assert_eq!(stage.prev_stage, None);
        assert_eq!(stage.next_stage, Some(1));
    }

    #[test]
    fn test_pipeline_stage_last() {
        let stage = PipelineStage::new(3, 4, vec![6, 7]);

        assert!(!stage.is_first());
        assert!(stage.is_last());
        assert_eq!(stage.prev_stage, Some(2));
        assert_eq!(stage.next_stage, None);
    }

    #[test]
    fn test_microbatch_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let mb = Microbatch::new(0, data.clone(), shape.clone());

        assert_eq!(mb.id, 0);
        assert_eq!(mb.data, data);
        assert_eq!(mb.shape, shape);
        assert_eq!(mb.current_stage, 0);
    }

    #[test]
    fn test_microbatch_advance() {
        let mut mb = Microbatch::new(0, vec![1.0], vec![1]);

        assert_eq!(mb.current_stage, 0);
        mb.advance_stage();
        assert_eq!(mb.current_stage, 1);
        mb.advance_stage();
        assert_eq!(mb.current_stage, 2);
    }

    #[test]
    fn test_activation_checkpointer_should_checkpoint() {
        let checkpointer = ActivationCheckpointer::new(2).expect("checkpointer creation failed");

        assert!(checkpointer.should_checkpoint(0));
        assert!(!checkpointer.should_checkpoint(1));
        assert!(checkpointer.should_checkpoint(2));
        assert!(!checkpointer.should_checkpoint(3));
        assert!(checkpointer.should_checkpoint(4));
    }

    #[tokio::test]
    async fn test_activation_checkpointer_store_retrieve() {
        let checkpointer = ActivationCheckpointer::new(1).expect("checkpointer creation failed");

        let activations = vec![1.0, 2.0, 3.0];
        checkpointer.checkpoint(0, activations.clone()).await;

        let retrieved = checkpointer.get_checkpoint(0).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("checkpoint retrieval failed"), activations);
    }

    #[tokio::test]
    async fn test_activation_checkpointer_clear() {
        let checkpointer = ActivationCheckpointer::new(1).expect("checkpointer creation failed");

        checkpointer.checkpoint(0, vec![1.0, 2.0]).await;
        checkpointer.checkpoint(1, vec![3.0, 4.0]).await;

        checkpointer.clear().await;

        assert_eq!(checkpointer.get_checkpoint(0).await, None);
        assert_eq!(checkpointer.get_checkpoint(1).await, None);
    }

    #[tokio::test]
    async fn test_activation_checkpointer_recomputation_count() {
        let checkpointer = ActivationCheckpointer::new(2).expect("checkpointer creation failed");

        assert_eq!(checkpointer.recomputation_count().await, 0);

        checkpointer.increment_recomputation().await;
        assert_eq!(checkpointer.recomputation_count().await, 1);

        checkpointer.increment_recomputation().await;
        assert_eq!(checkpointer.recomputation_count().await, 2);
    }

    #[test]
    fn test_activation_checkpointer_interval() {
        let checkpointer = ActivationCheckpointer::new(3).expect("checkpointer creation failed");

        assert_eq!(checkpointer.interval(), 3);
    }

    #[test]
    fn test_partition_strategy_equality() {
        assert_eq!(PartitionStrategy::ColumnWise, PartitionStrategy::ColumnWise);
        assert_ne!(PartitionStrategy::ColumnWise, PartitionStrategy::RowWise);
    }
}
