//! Efficient Communication Backends for Distributed Training
//!
//! This module provides high-performance communication primitives optimized for
//! distributed deep learning workloads. It builds on [`Communicator`]'s shared
//! [`super::net::Endpoint`] with specialized features for tensor communication
//! and bandwidth optimization.
//!
//! # Features
//!
//! - **Tensor Serialization**: Efficient serialization using oxicode
//! - **Async Primitives**: Non-blocking send/recv with async/await
//! - **Bandwidth Optimization**: Compression, batching, pipelining
//! - **Latency Hiding**: Computation/communication overlap
//! - **Topology-Aware**: Network-aware routing strategies
//!
//! # `isend`/`irecv` versus `send`/`recv`
//!
//! [`AsyncCommunicator::isend`]/[`AsyncCommunicator::send`] are the same
//! real, non-blocking-on-the-socket send ([`super::net::Endpoint::send_owned`]
//! already only awaits queue capacity, never the wire — see its docs), kept
//! as two names for API compatibility with callers written against the
//! historical isend/send split. [`AsyncCommunicator::irecv`] and
//! [`AsyncCommunicator::recv`] are genuinely different: `irecv` polls once
//! and returns immediately if nothing has arrived yet
//! ([`super::net::Endpoint::try_recv_bytes`]); `recv` waits up to the
//! endpoint's configured `recv_timeout`
//! ([`super::net::Endpoint::recv_bytes`]).
//!
//! All `AsyncCommunicator` traffic between two ranks travels under one fixed
//! wire tag (`ASYNC_COMM_TAG`) within the wrapped [`Communicator`]'s own
//! context — this API has no per-call tag parameter, so distinguishing
//! logical channels is left to [`TensorMessage::tag`] as user-level metadata
//! (matched by the application after receipt) rather than a transport-level
//! routing key.
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::communication::*;
//! use numrs2::distributed::process::*;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), CommunicationError> {
//! let world = init().await?;
//! let comm = AsyncCommunicator::new(Arc::new(world))?;
//!
//! // Send tensor with compression
//! let tensor = vec![1.0_f32; 1000];
//! let msg = TensorMessage::new(
//!     tensor,
//!     CompressionStrategy::TopK { k: 100 },
//!     MessagePriority::High
//! );
//! comm.isend(msg, 1).await?;
//!
//! // Receive tensor asynchronously
//! let received: TensorMessage<f32> = comm.irecv(0).await?;
//! # Ok(())
//! # }
//! ```

use super::net::SendOpts;
use super::process::{Communicator, ProcessError};
use crate::error::NumRs2Error;
use oxicode::{Decode, Encode};
use std::collections::VecDeque;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;

/// Errors that can occur during communication operations
#[derive(Error, Debug)]
pub enum CommunicationError {
    #[error("Process error: {0}")]
    Process(#[from] ProcessError),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Channel error: {0}")]
    Channel(String),

    #[error("Timeout: operation exceeded {0}ms")]
    Timeout(u64),

    #[error("Invalid rank {rank}, communicator size is {size}")]
    InvalidRank { rank: usize, size: usize },

    #[error("Message size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("Network error: {0}")]
    Network(String),
}

impl From<CommunicationError> for NumRs2Error {
    fn from(err: CommunicationError) -> Self {
        NumRs2Error::DistributedComputing(err.to_string())
    }
}

/// Message priority for prioritized communication.
///
/// Kept as message-level metadata (see the module docs): the underlying
/// [`super::net::Endpoint`] delivers strictly FIFO per `(src, ctx, tag)` key
/// with no cross-key reordering, so this no longer drives an actual send
/// order the way the old in-memory priority queue did. It remains useful for
/// an application to attach urgency information to a [`TensorMessage`] that
/// the receiving side can act on (e.g. process urgent messages first once
/// several have arrived).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Encode, Decode)]
pub enum MessagePriority {
    /// Low priority - background operations
    Low = 0,
    /// Normal priority - regular data transfers
    Normal = 1,
    /// High priority - critical synchronization
    High = 2,
    /// Urgent priority - control messages
    Urgent = 3,
}

/// Compression strategy for bandwidth optimization.
///
/// Definition and behavior moved to
/// [`super::compression`] — re-exported here so existing `communication::`
/// call sites keep working unchanged. `compress_tensor`/`decompress_tensor`
/// are re-exported the same way; see that module for the real
/// TopK/RandomK/Threshold/Quantization implementations and
/// [`super::compression::QuantizedTensor`] for bit-packed quantization.
pub use super::compression::CompressionStrategy;

/// Tensor message with metadata for efficient communication
#[derive(Debug, Clone, Encode, Decode)]
pub struct TensorMessage<T>
where
    T: Clone + Encode + Decode,
{
    /// Tensor data
    pub data: Vec<T>,

    /// Original shape of the tensor
    pub shape: Vec<usize>,

    /// Compression strategy used
    pub compression: CompressionStrategy,

    /// Message priority
    pub priority: MessagePriority,

    /// Sequence number for ordering
    pub sequence: u64,

    /// Sender rank
    pub sender: usize,

    /// Tag for message identification
    pub tag: u32,

    /// Indices for sparse tensors (used with compression)
    pub indices: Option<Vec<usize>>,
}

impl<T> TensorMessage<T>
where
    T: Clone + Encode + Decode,
{
    /// Create a new tensor message
    pub fn new(data: Vec<T>, compression: CompressionStrategy, priority: MessagePriority) -> Self {
        Self {
            shape: vec![data.len()],
            data,
            compression,
            priority,
            sequence: 0,
            sender: 0,
            tag: 0,
            indices: None,
        }
    }

    /// Create tensor message with shape
    pub fn with_shape(
        data: Vec<T>,
        shape: Vec<usize>,
        compression: CompressionStrategy,
        priority: MessagePriority,
    ) -> Self {
        Self {
            data,
            shape,
            compression,
            priority,
            sequence: 0,
            sender: 0,
            tag: 0,
            indices: None,
        }
    }

    /// Set sequence number
    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = sequence;
        self
    }

    /// Set sender rank
    pub fn with_sender(mut self, sender: usize) -> Self {
        self.sender = sender;
        self
    }

    /// Set message tag
    pub fn with_tag(mut self, tag: u32) -> Self {
        self.tag = tag;
        self
    }

    /// Get data size in bytes (uncompressed)
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<T>()
    }
}

/// Fixed wire tag every [`AsyncCommunicator`] send/recv uses. See the module
/// docs for why this API has no per-call tag parameter of its own, and
/// [`super::collective`]'s `TAG_*` constants for the (disjoint, far lower)
/// range the collective operations use on the same kind of communicator —
/// the two can never collide.
const ASYNC_COMM_TAG: u64 = 0xACC0_0000_0000_0001;

/// Asynchronous communicator with non-blocking operations, built directly on
/// a [`Communicator`]'s shared [`super::net::Endpoint`].
pub struct AsyncCommunicator {
    /// Underlying communicator
    communicator: Arc<Communicator>,

    /// Sequence counter for message ordering
    sequence_counter: Arc<Mutex<u64>>,
}

impl AsyncCommunicator {
    /// Create new async communicator
    pub fn new(communicator: Arc<Communicator>) -> Result<Self, CommunicationError> {
        Ok(Self {
            communicator,
            sequence_counter: Arc::new(Mutex::new(0)),
        })
    }

    /// Get next sequence number
    async fn next_sequence(&self) -> u64 {
        let mut counter = self.sequence_counter.lock().await;
        let seq = *counter;
        *counter += 1;
        seq
    }

    fn check_rank(&self, rank: usize) -> Result<(), CommunicationError> {
        if rank >= self.communicator.size() {
            return Err(CommunicationError::InvalidRank {
                rank,
                size: self.communicator.size(),
            });
        }
        Ok(())
    }

    /// Non-blocking send: encodes `message` and hands it to the shared
    /// endpoint, which only awaits queue capacity (never the socket) before
    /// returning. See the module docs for why this is effectively the same
    /// operation as [`Self::send`] under this transport.
    pub async fn isend<T>(
        &self,
        message: TensorMessage<T>,
        dest: usize,
    ) -> Result<(), CommunicationError>
    where
        T: Clone + Encode + Decode,
    {
        self.check_rank(dest)?;
        let data = oxicode::encode_to_vec(&message).map_err(|e| {
            CommunicationError::Serialization(format!("Failed to serialize message: {}", e))
        })?;
        let endpoint = self
            .communicator
            .require_endpoint()
            .map_err(CommunicationError::Process)?;
        let dst_global = self
            .communicator
            .global_rank(dest)
            .map_err(CommunicationError::Process)?;
        endpoint
            .send_owned(
                dst_global,
                self.communicator.context().0,
                ASYNC_COMM_TAG,
                data,
                SendOpts::default(),
            )
            .await
            .map_err(|e| CommunicationError::Network(e.to_string()))?;
        Ok(())
    }

    /// Non-blocking receive: returns immediately with
    /// [`CommunicationError::Channel`] if nothing has arrived yet, rather
    /// than waiting — see [`Self::recv`] for the blocking version.
    pub async fn irecv<T>(&self, source: usize) -> Result<TensorMessage<T>, CommunicationError>
    where
        T: Clone + Encode + Decode,
    {
        self.check_rank(source)?;
        let endpoint = self
            .communicator
            .require_endpoint()
            .map_err(CommunicationError::Process)?;
        let src_global = self
            .communicator
            .global_rank(source)
            .map_err(CommunicationError::Process)?;
        let data = endpoint
            .try_recv_bytes(src_global, self.communicator.context().0, ASYNC_COMM_TAG)
            .map_err(|e| CommunicationError::Network(e.to_string()))?
            .ok_or_else(|| {
                CommunicationError::Channel(format!("no data available yet from rank {source}"))
            })?;

        let (message, _) = oxicode::decode_from_slice(&data).map_err(|e| {
            CommunicationError::Deserialization(format!("Failed to deserialize message: {}", e))
        })?;

        Ok(message)
    }

    /// Blocking send (waits for completion) — see the module docs on why
    /// this and [`Self::isend`] are the same operation under this transport.
    pub async fn send<T>(
        &self,
        message: TensorMessage<T>,
        dest: usize,
    ) -> Result<(), CommunicationError>
    where
        T: Clone + Encode + Decode,
    {
        self.isend(message, dest).await
    }

    /// Blocking receive: waits up to the shared endpoint's configured
    /// `recv_timeout` for a message from `source`, unlike [`Self::irecv`]
    /// which never waits.
    pub async fn recv<T>(&self, source: usize) -> Result<TensorMessage<T>, CommunicationError>
    where
        T: Clone + Encode + Decode,
    {
        self.check_rank(source)?;
        let endpoint = self
            .communicator
            .require_endpoint()
            .map_err(CommunicationError::Process)?;
        let src_global = self
            .communicator
            .global_rank(source)
            .map_err(CommunicationError::Process)?;
        let data = endpoint
            .recv_bytes(src_global, self.communicator.context().0, ASYNC_COMM_TAG)
            .await
            .map_err(|e| CommunicationError::Network(e.to_string()))?;

        let (message, _) = oxicode::decode_from_slice(&data).map_err(|e| {
            CommunicationError::Deserialization(format!("Failed to deserialize message: {}", e))
        })?;

        Ok(message)
    }

    /// Get communicator rank
    pub fn rank(&self) -> usize {
        self.communicator.rank()
    }

    /// Get communicator size
    pub fn size(&self) -> usize {
        self.communicator.size()
    }
}

/// Pipelined communicator for latency hiding
pub struct PipelinedCommunicator {
    /// Base async communicator
    base: AsyncCommunicator,

    /// Pipeline depth (number of concurrent operations)
    depth: usize,

    /// Active pipeline stages
    active_stages: Arc<Mutex<VecDeque<PipelineStage>>>,
}

#[derive(Debug)]
struct PipelineStage {
    operation_id: u64,
    status: PipelineStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineStatus {
    Pending,
    // Reserved for a future in-flight state between `Pending` and
    // `Completed`; no code transitions a stage into it yet (today a stage
    // is only ever `Pending` until something removes it), so it is never
    // constructed. Kept as a documented part of the intended state machine
    // rather than deleted.
    #[allow(dead_code)]
    InProgress,
    Completed,
}

impl PipelinedCommunicator {
    /// Create new pipelined communicator
    pub fn new(communicator: Arc<Communicator>, depth: usize) -> Result<Self, CommunicationError> {
        Ok(Self {
            base: AsyncCommunicator::new(communicator)?,
            depth,
            active_stages: Arc::new(Mutex::new(VecDeque::new())),
        })
    }

    /// Start a pipelined send operation
    pub async fn pipeline_send<T>(
        &self,
        message: TensorMessage<T>,
        dest: usize,
    ) -> Result<u64, CommunicationError>
    where
        T: Clone + Encode + Decode,
    {
        // Wait if pipeline is full
        self.wait_for_pipeline_slot().await?;

        // Get operation ID
        let op_id = self.base.next_sequence().await;

        // Add to pipeline
        let mut stages = self.active_stages.lock().await;
        stages.push_back(PipelineStage {
            operation_id: op_id,
            status: PipelineStatus::Pending,
        });
        drop(stages);

        // Start the real (enqueue-and-return) send.
        self.base.isend(message, dest).await?;

        Ok(op_id)
    }

    /// Wait for a specific pipeline operation to complete
    pub async fn wait_operation(&self, op_id: u64) -> Result<(), CommunicationError> {
        let mut stages = self.active_stages.lock().await;

        // Find and remove the operation
        let pos = stages.iter().position(|s| s.operation_id == op_id);
        if let Some(pos) = pos {
            stages.remove(pos);
        }

        Ok(())
    }

    /// Wait for all pipeline operations to complete
    pub async fn wait_all(&self) -> Result<(), CommunicationError> {
        let mut stages = self.active_stages.lock().await;
        stages.clear();
        Ok(())
    }

    /// Wait for pipeline slot to become available
    async fn wait_for_pipeline_slot(&self) -> Result<(), CommunicationError> {
        loop {
            let mut stages = self.active_stages.lock().await;
            if stages.len() < self.depth {
                return Ok(());
            }

            // Remove completed stages
            while let Some(stage) = stages.front() {
                if stage.status == PipelineStatus::Completed {
                    stages.pop_front();
                } else {
                    break;
                }
            }

            drop(stages);

            // Small delay before checking again
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }
    }

    /// Get pipeline depth
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Get number of active operations
    pub async fn active_count(&self) -> usize {
        self.active_stages.lock().await.len()
    }
}

// `compress_tensor`/`decompress_tensor` implementations moved to
// `super::compression` (bound relaxed to `Float`, TopK now selects by
// absolute value, RandomK does real seeded sampling, Threshold actually
// thresholds, Quantization returns a clear error pointing at
// `compression::QuantizedTensor`). Re-exported here for existing call sites.
pub use super::compression::{compress_tensor, decompress_tensor};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::process::Communicator;
    use crate::distributed::testing::{ClusterNode, LocalCluster};
    use std::time::Duration;

    #[test]
    fn test_tensor_message_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let msg = TensorMessage::new(
            data.clone(),
            CompressionStrategy::None,
            MessagePriority::Normal,
        );

        assert_eq!(msg.data, data);
        assert_eq!(msg.shape, vec![4]);
        assert_eq!(msg.priority, MessagePriority::Normal);
    }

    #[test]
    fn test_tensor_message_with_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let msg = TensorMessage::with_shape(
            data.clone(),
            shape.clone(),
            CompressionStrategy::None,
            MessagePriority::High,
        );

        assert_eq!(msg.data, data);
        assert_eq!(msg.shape, shape);
        assert_eq!(msg.priority, MessagePriority::High);
    }

    #[test]
    fn test_message_priority_ordering() {
        assert!(MessagePriority::Urgent > MessagePriority::High);
        assert!(MessagePriority::High > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
    }

    // Compression/decompression behavior tests (compress_tensor,
    // decompress_tensor, CompressionStrategy serialization, QuantizedTensor)
    // live in `super::compression`, which now owns those implementations.

    #[test]
    fn test_tensor_message_serialization() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let msg = TensorMessage::new(data, CompressionStrategy::None, MessagePriority::Normal);

        let serialized = oxicode::encode_to_vec(&msg);
        assert!(serialized.is_ok());

        let bytes = serialized.expect("serialization failed");
        let deserialized: Result<(TensorMessage<f32>, usize), _> =
            oxicode::decode_from_slice(&bytes);
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_message_priority_serialization() {
        let priorities = vec![
            MessagePriority::Low,
            MessagePriority::Normal,
            MessagePriority::High,
            MessagePriority::Urgent,
        ];

        for priority in priorities {
            let serialized = oxicode::encode_to_vec(&priority);
            assert!(serialized.is_ok());

            let bytes = serialized.expect("serialization failed");
            let deserialized: Result<(MessagePriority, usize), _> =
                oxicode::decode_from_slice(&bytes);
            assert!(deserialized.is_ok());
        }
    }

    #[test]
    fn test_tensor_message_size() {
        let data = vec![1.0_f64; 1000];
        let msg = TensorMessage::new(data, CompressionStrategy::None, MessagePriority::Normal);

        let size = msg.size_bytes();
        assert_eq!(size, 1000 * std::mem::size_of::<f64>());
    }

    fn short_timeout_config() -> super::super::net::EndpointConfig {
        super::super::net::EndpointConfig {
            recv_timeout: Duration::from_secs(2),
            ..super::super::net::EndpointConfig::default()
        }
    }

    /// `irecv` must be truly non-blocking: nothing has been sent yet, so it
    /// must fail immediately rather than hang.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn irecv_returns_immediately_when_nothing_has_arrived() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            2,
            cfg,
            Duration::from_secs(10),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(std::sync::Arc::new(node.endpoint))?;
                let async_comm = AsyncCommunicator::new(std::sync::Arc::new(comm))
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                let other = if async_comm.rank() == 0 { 1 } else { 0 };
                let result: Result<TensorMessage<f32>, _> = async_comm.irecv(other).await;
                Ok(result.is_err())
            },
        )
        .await
        .expect("run");
        assert!(results.iter().all(|&was_err| was_err));
    }

    /// A real send/recv round trip: rank 0 sends a tensor to rank 1, which
    /// receives it (blocking `recv`, since it may need to wait a moment for
    /// the send to land) and checks the payload survived intact.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn isend_and_recv_round_trip_a_real_tensor() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            2,
            cfg,
            Duration::from_secs(10),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(std::sync::Arc::new(node.endpoint))?;
                let rank = comm.rank();
                let async_comm = AsyncCommunicator::new(std::sync::Arc::new(comm))
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                if rank == 0 {
                    let msg = TensorMessage::new(
                        vec![1.0_f32, 2.0, 3.0, 4.0],
                        CompressionStrategy::None,
                        MessagePriority::High,
                    );
                    async_comm
                        .isend(msg, 1)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    Ok(Vec::new())
                } else {
                    let got: TensorMessage<f32> = async_comm
                        .recv(0)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    Ok(got.data)
                }
            },
        )
        .await
        .expect("run");
        assert_eq!(results[1], vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// `PipelinedCommunicator` sends must also be real: this drives
    /// `pipeline_send` end to end and confirms the payload actually arrives.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pipelined_send_delivers_a_real_message() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            2,
            cfg,
            Duration::from_secs(10),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(std::sync::Arc::new(node.endpoint))?;
                let rank = comm.rank();
                let pipeline = PipelinedCommunicator::new(std::sync::Arc::new(comm), 4)
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                if rank == 0 {
                    let msg = TensorMessage::new(
                        vec![9_i32, 8, 7],
                        CompressionStrategy::None,
                        MessagePriority::Normal,
                    );
                    let op_id = pipeline
                        .pipeline_send(msg, 1)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    pipeline
                        .wait_operation(op_id)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    Ok(Vec::new())
                } else {
                    let got: TensorMessage<i32> = pipeline
                        .base
                        .recv(0)
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    Ok(got.data)
                }
            },
        )
        .await
        .expect("run");
        assert_eq!(results[1], vec![9, 8, 7]);
    }
}
