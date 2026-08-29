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

use super::communication::CommunicationError;
use super::coordinator::CoordinatorError;
use super::process::{Communicator, ProcessError};
use crate::error::NumRs2Error;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

    /// A failure from the real [`super::net`] transport layer, surfaced by
    /// [`PipelineParallel`]'s send/recv methods going through
    /// [`super::net::endpoint::Endpoint`] directly.
    #[error("Transport error: {0}")]
    Net(#[from] super::net::NetError),

    /// A failure from a real [`super::collective`] operation, surfaced by
    /// [`TensorParallel::gather`].
    #[error("Collective operation error: {0}")]
    Collective(#[from] super::collective::CollectiveError),

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

/// Base wire tag for forward-activation transfers in [`PipelineParallel`];
/// folds in the *sending* stage's own id and the microbatch id so every
/// (originating stage, microbatch) pair gets its own tag — this is what lets
/// [`PipelineParallel::recv_forward`] pull a specific microbatch's
/// activations regardless of what order other microbatches for the same
/// stage-pair arrive in, rather than being limited to strict FIFO receipt
/// (the "irecv-style" point-to-point the task calls for). See
/// `pipeline_tag` for the exact bit layout, and [`super::optimization`]'s
/// module docs for the crate-wide convention keeping every tag range in this
/// file disjoint from collectives (`0x1..0xA`), the latency/bandwidth probes
/// (`0xD`/`0xE`), and [`super::communication::AsyncCommunicator`]'s
/// `ASYNC_COMM_TAG` (`0xACC0_...`).
const TAG_PIPELINE_FORWARD_BASE: u64 = 0xB_0000_0000;

/// As [`TAG_PIPELINE_FORWARD_BASE`], for backward-gradient transfers.
const TAG_PIPELINE_BACKWARD_BASE: u64 = 0xC_0000_0000;

/// Fold `stage_id` (the *sending* stage's own id — known directly by the
/// sender, and recoverable by the receiver from its own `prev_stage`/
/// `next_stage`) and `microbatch_id` into an offset from `base`, giving
/// every `(base, stage_id, microbatch_id)` combination its own wire tag.
/// `stage_id` occupies bits 24..31 of the offset (up to 255 stages) and
/// `microbatch_id` occupies bits 0..23 (up to ~16.7 million microbatches) —
/// both far beyond any realistic pipeline configuration — so neither can
/// collide with the other, and the combined offset (always `< 2^32`) never
/// reaches into the tag range above `base`.
fn pipeline_tag(base: u64, stage_id: usize, microbatch_id: usize) -> u64 {
    let stage_part = (stage_id as u64 & 0xFF) << 24;
    let microbatch_part = microbatch_id as u64 & 0x00FF_FFFF;
    base + stage_part + microbatch_part
}

/// Encode an `f32` tensor for the wire, the same way every other tensor
/// payload in this crate does.
fn encode_tensor(data: &[f32]) -> Result<Vec<u8>, ModelParallelError> {
    oxicode::encode_to_vec(&data.to_vec())
        .map_err(|e| ModelParallelError::PipelineError(format!("failed to encode tensor: {e}")))
}

/// Inverse of [`encode_tensor`].
fn decode_tensor(bytes: &[u8]) -> Result<Vec<f32>, ModelParallelError> {
    let (data, _): (Vec<f32>, usize) = oxicode::decode_from_slice(bytes)
        .map_err(|e| ModelParallelError::PipelineError(format!("failed to decode tensor: {e}")))?;
    Ok(data)
}

/// Pipeline parallel coordinator.
///
/// [`Self::send_forward`]/[`Self::recv_forward`]/[`Self::send_backward`]/
/// [`Self::recv_backward`] are real point-to-point transfers over
/// `communicator`'s shared [`super::net::endpoint::Endpoint`], each under a
/// tag computed by `pipeline_tag` that is unique to its
/// `(direction, originating stage, microbatch_id)` — see that function's
/// docs. `recv_forward`/`recv_backward` wait on
/// [`super::net::endpoint::Endpoint::recv_bytes`] for the exact microbatch
/// requested (an earlier version of this type instead always returned a
/// fixed 10-element placeholder for anything not already sitting in a local,
/// never-actually-populated buffer — every call silently fabricated a
/// result regardless of what, if anything, had really been sent).
pub struct PipelineParallel {
    /// Communicator
    communicator: Arc<Communicator>,

    /// This worker's pipeline stage
    stage: PipelineStage,

    /// Number of ranks assigned to each stage (used to translate a stage id
    /// into the local rank of that stage's first member — see
    /// [`Self::rank_for_stage`]).
    ranks_per_stage: usize,

    /// Number of microbatches
    num_microbatches: usize,

    /// Pipeline schedule
    // Always `PipelineSchedule::GPipe` today (see `new`) and never read back:
    // `send_forward`/`recv_forward`/etc. don't branch on it, so both this
    // field and `PipelineSchedule::OneFOneB` document an intended future
    // scheduling knob rather than dead leftovers. Real 1F1B interleaving is
    // a scheduling-behavior change, out of scope for a lint-only pass.
    #[allow(dead_code)]
    schedule: PipelineSchedule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineSchedule {
    /// GPipe: forward all microbatches, then backward all
    GPipe,

    /// PipeDream: interleaved 1F1B (one forward, one backward)
    // Never constructed: `PipelineParallel::new` always picks `GPipe` (see
    // the `schedule` field's own comment). Kept named and documented as the
    // scheduling mode a real 1F1B implementation would select.
    #[allow(dead_code)]
    OneFOneB,
}

impl PipelineParallel {
    /// Create new pipeline parallel coordinator
    pub fn new(
        communicator: Arc<Communicator>,
        num_stages: usize,
        num_microbatches: usize,
    ) -> Result<Self, ModelParallelError> {
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
            stage,
            ranks_per_stage,
            num_microbatches,
            schedule: PipelineSchedule::GPipe,
        })
    }

    /// The (local) rank of `stage_id`'s first member — the single
    /// point-to-point target this type's send/recv methods use to represent
    /// "the next/previous stage" (consistent with how [`PipelineStage`]
    /// itself only tracks stage-level, not rank-level, adjacency).
    fn rank_for_stage(&self, stage_id: usize) -> usize {
        stage_id * self.ranks_per_stage
    }

    /// Send forward activations to next stage
    pub async fn send_forward(
        &self,
        microbatch_id: usize,
        activations: &[f32],
    ) -> Result<(), ModelParallelError> {
        if let Some(next_stage) = self.stage.next_stage {
            let next_rank_local = self.rank_for_stage(next_stage);
            let endpoint = self.communicator.require_endpoint()?;
            let next_rank_global = self.communicator.global_rank(next_rank_local)?;
            let tag = pipeline_tag(
                TAG_PIPELINE_FORWARD_BASE,
                self.stage.stage_id,
                microbatch_id,
            );
            let payload = encode_tensor(activations)?;
            endpoint
                .send_owned(
                    next_rank_global,
                    self.communicator.context().0,
                    tag,
                    payload,
                    super::net::SendOpts::default(),
                )
                .await?;
        }

        Ok(())
    }

    /// Receive forward activations from previous stage: waits for
    /// specifically `microbatch_id`'s activations (see `pipeline_tag`),
    /// not merely the next message to arrive from that stage.
    pub async fn recv_forward(&self, microbatch_id: usize) -> Result<Vec<f32>, ModelParallelError> {
        if let Some(prev_stage) = self.stage.prev_stage {
            let prev_rank_local = self.rank_for_stage(prev_stage);
            let endpoint = self.communicator.require_endpoint()?;
            let prev_rank_global = self.communicator.global_rank(prev_rank_local)?;
            let tag = pipeline_tag(TAG_PIPELINE_FORWARD_BASE, prev_stage, microbatch_id);
            let bytes = endpoint
                .recv_bytes(prev_rank_global, self.communicator.context().0, tag)
                .await?;
            decode_tensor(&bytes)
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
            let prev_rank_local = self.rank_for_stage(prev_stage);
            let endpoint = self.communicator.require_endpoint()?;
            let prev_rank_global = self.communicator.global_rank(prev_rank_local)?;
            let tag = pipeline_tag(
                TAG_PIPELINE_BACKWARD_BASE,
                self.stage.stage_id,
                microbatch_id,
            );
            let payload = encode_tensor(gradients)?;
            endpoint
                .send_owned(
                    prev_rank_global,
                    self.communicator.context().0,
                    tag,
                    payload,
                    super::net::SendOpts::default(),
                )
                .await?;
        }

        Ok(())
    }

    /// Receive backward gradients from next stage: waits for specifically
    /// `microbatch_id`'s gradients (see `pipeline_tag`), not merely the
    /// next message to arrive from that stage.
    pub async fn recv_backward(
        &self,
        microbatch_id: usize,
    ) -> Result<Vec<f32>, ModelParallelError> {
        if let Some(next_stage) = self.stage.next_stage {
            let next_rank_local = self.rank_for_stage(next_stage);
            let endpoint = self.communicator.require_endpoint()?;
            let next_rank_global = self.communicator.global_rank(next_rank_local)?;
            let tag = pipeline_tag(TAG_PIPELINE_BACKWARD_BASE, next_stage, microbatch_id);
            let bytes = endpoint
                .recv_bytes(next_rank_global, self.communicator.context().0, tag)
                .await?;
            decode_tensor(&bytes)
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

/// Tensor parallel coordinator.
///
/// [`Self::gather`] delegates to [`super::collective::allgather`] over the
/// real transport — an earlier version instead always returned
/// `local_tensor` unchanged, silently skipping the all-gather rather than
/// actually reassembling every rank's partition into the full tensor.
pub struct TensorParallel {
    /// Communicator
    communicator: Arc<Communicator>,

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
        let tp_size = communicator.size();
        let tp_rank = communicator.rank();

        Ok(Self {
            communicator,
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

    /// All-gather this rank's partition together with every other rank's,
    /// via [`super::collective::allgather`] (real network traffic, not a
    /// same-rank echo). The concatenation order matches rank order — the
    /// inverse of [`Self::partition`]'s `ColumnWise`/`RowWise`/`BatchWise`/
    /// `SequenceWise` split only for the block-contiguous strategies
    /// (`ColumnWise`'s column-major partitions do not literally concatenate
    /// back into row-major original order; reassembling that shape is left
    /// to the caller, same as before this rewrite).
    pub async fn gather(&self, local_tensor: &[f32]) -> Result<Vec<f32>, ModelParallelError> {
        Ok(super::collective::allgather(local_tensor, &self.communicator).await?)
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

    // -----------------------------------------------------------------
    // pipeline_tag
    // -----------------------------------------------------------------

    #[test]
    fn pipeline_tag_is_distinct_per_stage_and_microbatch() {
        let base = TAG_PIPELINE_FORWARD_BASE;
        // Different stage, same microbatch.
        assert_ne!(pipeline_tag(base, 0, 5), pipeline_tag(base, 1, 5));
        // Same stage, different microbatch.
        assert_ne!(pipeline_tag(base, 2, 0), pipeline_tag(base, 2, 1));
        // Forward and backward bases never collide for any stage/microbatch.
        assert_ne!(
            pipeline_tag(TAG_PIPELINE_FORWARD_BASE, 3, 7),
            pipeline_tag(TAG_PIPELINE_BACKWARD_BASE, 3, 7)
        );
    }

    // -----------------------------------------------------------------
    // PipelineParallel: real point-to-point forward/backward exchange
    // over a real 2-rank LocalCluster (p=2, 2 pipeline stages).
    // -----------------------------------------------------------------

    use crate::distributed::testing::{ClusterNode, LocalCluster};
    use std::time::Duration;

    fn short_timeout_config() -> super::super::net::EndpointConfig {
        super::super::net::EndpointConfig {
            recv_timeout: Duration::from_secs(5),
            ..super::super::net::EndpointConfig::default()
        }
    }

    /// A toy full forward+backward exchange across a real 2-process,
    /// 2-stage pipeline: rank 0 (stage 0) sends forward activations to rank
    /// 1 (stage 1), which replies with backward gradients; both directions
    /// are verified to have delivered the real payload sent, not a
    /// fabricated placeholder (an earlier version of `recv_forward`/
    /// `recv_backward` always returned a fixed 10-element zero vector for
    /// anything not already sitting in a buffer nothing ever populated).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pipeline_forward_and_backward_round_trip_p2() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            2,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let pipeline = PipelineParallel::new(Arc::new(comm.clone()), 2, 1)
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;

                if pipeline.stage().is_first() {
                    // Stage 0: send a real forward activation, then wait for
                    // stage 1's real backward gradient reply.
                    let activations = vec![1.0_f32, 2.0, 3.0, 4.0];
                    pipeline
                        .send_forward(0, &activations)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    let gradients = pipeline
                        .recv_backward(0)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    Ok(gradients)
                } else {
                    // Stage 1: receive the real forward activation, verify
                    // it, then send a real backward gradient back.
                    let received = pipeline
                        .recv_forward(0)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    assert_eq!(received, vec![1.0, 2.0, 3.0, 4.0]);
                    let gradients = vec![0.1_f32, 0.2, 0.3, 0.4];
                    pipeline
                        .send_backward(0, &gradients)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    Ok(Vec::new())
                }
            },
        )
        .await
        .expect("pipeline forward/backward run");

        // Rank 0 (stage 0) is the one that returns the real gradients it
        // received back from stage 1.
        assert_eq!(results[0], vec![0.1, 0.2, 0.3, 0.4]);
    }

    /// `recv_forward` at the first stage (no previous stage) and
    /// `recv_backward` at the last stage (no next stage) must fail cleanly
    /// rather than fabricate a placeholder result.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pipeline_recv_at_pipeline_ends_is_an_explicit_error() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            2,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let pipeline = PipelineParallel::new(Arc::new(comm.clone()), 2, 1)
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;

                let is_err = if pipeline.stage().is_first() {
                    pipeline.recv_forward(0).await.is_err()
                } else {
                    pipeline.recv_backward(0).await.is_err()
                };
                Ok(is_err)
            },
        )
        .await
        .expect("pipeline end-error run");

        assert!(
            results.iter().all(|&is_err| is_err),
            "recv at a pipeline end without a corresponding neighbor stage must error"
        );
    }

    /// A microbatch id distinct from the one sent must not be delivered:
    /// `recv_forward(1)` should not spuriously receive a payload sent under
    /// `send_forward(0, ...)`. Uses `try_recv`-style reasoning via a short
    /// per-recv timeout so a wrongly-matched delivery would show up as a
    /// wrong value rather than an indefinite hang.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pipeline_forward_is_scoped_to_its_own_microbatch_id() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            2,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let pipeline = PipelineParallel::new(Arc::new(comm.clone()), 2, 2)
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;

                if pipeline.stage().is_first() {
                    pipeline
                        .send_forward(0, &[10.0, 20.0])
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    pipeline
                        .send_forward(1, &[30.0, 40.0])
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    Ok(Vec::new())
                } else {
                    // Deliberately receive microbatch 1 first: this must
                    // pick up {30, 40}, not microbatch 0's {10, 20}, proving
                    // the tag genuinely scopes delivery by microbatch id
                    // rather than plain FIFO order.
                    let mb1 = pipeline
                        .recv_forward(1)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    let mb0 = pipeline
                        .recv_forward(0)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    Ok([mb1, mb0].concat())
                }
            },
        )
        .await
        .expect("microbatch scoping run");

        assert_eq!(results[1], vec![30.0, 40.0, 10.0, 20.0]);
    }

    // -----------------------------------------------------------------
    // TensorParallel::gather: real all-gather over a real LocalCluster.
    // -----------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tensor_parallel_gather_collects_every_rank_over_a_real_cluster() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            3,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let tp = TensorParallel::new(Arc::new(comm.clone()), PartitionStrategy::BatchWise)
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                let local = vec![(comm.rank() as f32 + 1.0) * 10.0];
                let gathered = tp
                    .gather(&local)
                    .await
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                Ok(gathered)
            },
        )
        .await
        .expect("tensor parallel gather run");

        for got in results {
            assert_eq!(got, vec![10.0, 20.0, 30.0]);
        }
    }
}
