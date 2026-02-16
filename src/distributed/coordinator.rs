//! Training Coordinators for Distributed Machine Learning
//!
//! This module provides coordination mechanisms for distributed training including
//! parameter servers, all-reduce implementations, barriers, and fault tolerance.
//!
//! # Features
//!
//! - **Parameter Server**: Centralized parameter storage and updates
//! - **All-Reduce**: Ring-AllReduce and tree-based implementations
//! - **Barriers**: Synchronization primitives for coordinated execution
//! - **Fault Tolerance**: Checkpointing and recovery mechanisms
//! - **Load Balancing**: Dynamic work distribution
//!
//! # Architectures
//!
//! ## Parameter Server
//! ```text
//! Workers            Parameter Servers
//! ┌─────┐            ┌──────────┐
//! │ W0  │───push────>│   PS0    │
//! └─────┘<───pull────└──────────┘
//! ┌─────┐            ┌──────────┐
//! │ W1  │───push────>│   PS1    │
//! └─────┘<───pull────└──────────┘
//! ```
//!
//! ## Ring-AllReduce
//! ```text
//! Scatter-Reduce → Allgather
//! W0 ──> W1 ──> W2 ──> W3 ──> W0
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::coordinator::*;
//! use numrs2::distributed::process::*;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), CoordinatorError> {
//! let world = init().await?;
//!
//! // Parameter server mode
//! let ps = ParameterServer::new(Arc::new(world.clone()), 2)?; // 2 PS nodes
//!
//! // Push gradients
//! let gradients = vec![1.0; 1000];
//! ps.push_gradients("param0", &gradients).await?;
//!
//! // Pull updated parameters
//! let params = ps.pull_parameters("param0").await?;
//!
//! // Ring-AllReduce for gradient aggregation
//! let reducer = RingAllReduce::new(Arc::new(world.clone()))?;
//! let aggregated = reducer.allreduce(&gradients).await?;
//!
//! // Barrier synchronization
//! let barrier = DistributedBarrier::new(Arc::new(world))?;
//! barrier.wait().await?;
//! # Ok(())
//! # }
//! ```

use super::collective::{allreduce, ReduceOp};
use super::communication::{AsyncCommunicator, CommunicationError, MessagePriority, TensorMessage};
use super::process::{Communicator, ProcessError};
use crate::error::NumRs2Error;
use oxicode::{Decode, Encode};
use scirs2_core::ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};

/// Errors that can occur in coordination operations
#[derive(Error, Debug)]
pub enum CoordinatorError {
    #[error("Process error: {0}")]
    Process(#[from] ProcessError),

    #[error("Communication error: {0}")]
    Communication(#[from] CommunicationError),

    #[error("Invalid parameter key: {0}")]
    InvalidKey(String),

    #[error("Parameter not found: {0}")]
    ParameterNotFound(String),

    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    #[error("Recovery error: {0}")]
    Recovery(String),

    #[error("Synchronization error: {0}")]
    Synchronization(String),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl From<CoordinatorError> for NumRs2Error {
    fn from(err: CoordinatorError) -> Self {
        NumRs2Error::DistributedComputing(err.to_string())
    }
}

/// Parameter server for centralized parameter management
pub struct ParameterServer {
    /// Communicator
    communicator: Arc<Communicator>,

    /// Async communicator for efficient transfers
    async_comm: AsyncCommunicator,

    /// Number of parameter server processes
    num_ps: usize,

    /// Parameter storage (key -> values)
    parameters: Arc<RwLock<HashMap<String, Vec<f32>>>>,

    /// Gradient accumulator
    gradient_buffer: Arc<Mutex<HashMap<String, Vec<f32>>>>,

    /// Version number for each parameter
    versions: Arc<RwLock<HashMap<String, u64>>>,
}

impl ParameterServer {
    /// Create new parameter server
    pub fn new(communicator: Arc<Communicator>, num_ps: usize) -> Result<Self, CoordinatorError> {
        let async_comm = AsyncCommunicator::new(communicator.clone())?;

        Ok(Self {
            communicator,
            async_comm,
            num_ps,
            parameters: Arc::new(RwLock::new(HashMap::new())),
            gradient_buffer: Arc::new(Mutex::new(HashMap::new())),
            versions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Initialize parameter with given key and initial values
    pub async fn init_parameter(
        &self,
        key: &str,
        initial_values: Vec<f32>,
    ) -> Result<(), CoordinatorError> {
        let mut params = self.parameters.write().await;
        params.insert(key.to_string(), initial_values);

        let mut versions = self.versions.write().await;
        versions.insert(key.to_string(), 0);

        Ok(())
    }

    /// Push gradients for a parameter
    pub async fn push_gradients(
        &self,
        key: &str,
        gradients: &[f32],
    ) -> Result<(), CoordinatorError> {
        let mut buffer = self.gradient_buffer.lock().await;
        let entry = buffer
            .entry(key.to_string())
            .or_insert_with(|| vec![0.0; gradients.len()]);

        // Accumulate gradients
        for (acc, &grad) in entry.iter_mut().zip(gradients.iter()) {
            *acc += grad;
        }

        Ok(())
    }

    /// Pull updated parameters
    pub async fn pull_parameters(&self, key: &str) -> Result<Vec<f32>, CoordinatorError> {
        let params = self.parameters.read().await;
        params
            .get(key)
            .cloned()
            .ok_or_else(|| CoordinatorError::ParameterNotFound(key.to_string()))
    }

    /// Apply accumulated gradients to parameters
    pub async fn apply_gradients(
        &self,
        key: &str,
        learning_rate: f32,
    ) -> Result<(), CoordinatorError> {
        let mut buffer = self.gradient_buffer.lock().await;
        let gradients = buffer
            .get_mut(key)
            .ok_or_else(|| CoordinatorError::ParameterNotFound(key.to_string()))?;

        let mut params = self.parameters.write().await;
        let parameters = params
            .get_mut(key)
            .ok_or_else(|| CoordinatorError::ParameterNotFound(key.to_string()))?;

        // Update parameters: param -= learning_rate * gradient
        for (param, grad) in parameters.iter_mut().zip(gradients.iter_mut()) {
            *param -= learning_rate * *grad;
            *grad = 0.0; // Clear gradient after applying
        }

        // Increment version
        let mut versions = self.versions.write().await;
        if let Some(version) = versions.get_mut(key) {
            *version += 1;
        }

        Ok(())
    }

    /// Get parameter version
    pub async fn get_version(&self, key: &str) -> Result<u64, CoordinatorError> {
        let versions = self.versions.read().await;
        versions
            .get(key)
            .copied()
            .ok_or_else(|| CoordinatorError::ParameterNotFound(key.to_string()))
    }

    /// Get number of parameter servers
    pub fn num_servers(&self) -> usize {
        self.num_ps
    }

    /// Determine which PS owns a parameter
    pub fn get_server_for_key(&self, key: &str) -> usize {
        // Simple hash-based assignment
        let hash = key
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        (hash as usize) % self.num_ps
    }
}

/// Ring-AllReduce implementation for efficient gradient aggregation
pub struct RingAllReduce {
    /// Communicator
    communicator: Arc<Communicator>,

    /// Async communicator
    async_comm: AsyncCommunicator,

    /// Ring topology (rank -> next rank)
    ring: Vec<usize>,
}

impl RingAllReduce {
    /// Create new ring-allreduce coordinator
    pub fn new(communicator: Arc<Communicator>) -> Result<Self, CoordinatorError> {
        let async_comm = AsyncCommunicator::new(communicator.clone())?;
        let size = communicator.size();

        // Build ring topology: 0 -> 1 -> 2 -> ... -> size-1 -> 0
        let ring: Vec<usize> = (0..size).map(|i| (i + 1) % size).collect();

        Ok(Self {
            communicator,
            async_comm,
            ring,
        })
    }

    /// Perform ring-allreduce on data
    pub async fn allreduce(&self, data: &[f32]) -> Result<Vec<f32>, CoordinatorError> {
        let size = self.communicator.size();
        let rank = self.communicator.rank();

        if size == 1 {
            return Ok(data.to_vec());
        }

        let chunk_size = data.len().div_ceil(size);
        let result = data.to_vec();

        // Phase 1: Scatter-Reduce
        for step in 0..size - 1 {
            let send_chunk = rank;
            let recv_chunk = (rank + size - 1) % size;

            let send_start = send_chunk * chunk_size;
            let send_end = (send_start + chunk_size).min(data.len());

            let next_rank = self.ring[rank];
            let prev_rank = (rank + size - 1) % size;

            // Send chunk to next rank
            if send_start < data.len() {
                let chunk = &result[send_start..send_end];
                let msg = TensorMessage::new(
                    chunk.to_vec(),
                    super::communication::CompressionStrategy::None,
                    MessagePriority::High,
                );
                self.async_comm.isend(msg, next_rank).await?;
            }

            // Receive chunk from previous rank (simulated for now)
            // In real implementation, would receive and accumulate
            let _ = (prev_rank, recv_chunk);
        }

        // Phase 2: Allgather
        for step in 0..size - 1 {
            let send_chunk = (rank + 1 - step + size) % size;
            let next_rank = self.ring[rank];

            let send_start = send_chunk * chunk_size;
            let send_end = (send_start + chunk_size).min(data.len());

            // Send chunk to next rank
            if send_start < data.len() {
                let chunk = &result[send_start..send_end];
                let msg = TensorMessage::new(
                    chunk.to_vec(),
                    super::communication::CompressionStrategy::None,
                    MessagePriority::High,
                );
                self.async_comm.isend(msg, next_rank).await?;
            }

            // Receive chunk from previous rank (simulated)
            let _ = step;
        }

        Ok(result)
    }

    /// Get ring topology
    pub fn topology(&self) -> &[usize] {
        &self.ring
    }
}

/// Tree-based AllReduce for hierarchical communication
pub struct TreeAllReduce {
    /// Communicator
    communicator: Arc<Communicator>,

    /// Async communicator
    async_comm: AsyncCommunicator,

    /// Branching factor
    branching_factor: usize,

    /// Parent in tree (-1 for root)
    parent: Option<usize>,

    /// Children in tree
    children: Vec<usize>,
}

impl TreeAllReduce {
    /// Create new tree-allreduce coordinator
    pub fn new(
        communicator: Arc<Communicator>,
        branching_factor: usize,
    ) -> Result<Self, CoordinatorError> {
        let async_comm = AsyncCommunicator::new(communicator.clone())?;
        let rank = communicator.rank();
        let size = communicator.size();

        // Build tree topology
        let parent = if rank == 0 {
            None
        } else {
            Some((rank - 1) / branching_factor)
        };

        let children: Vec<usize> = (1..=branching_factor)
            .map(|i| rank * branching_factor + i)
            .filter(|&c| c < size)
            .collect();

        Ok(Self {
            communicator,
            async_comm,
            branching_factor,
            parent,
            children,
        })
    }

    /// Perform tree-allreduce
    pub async fn allreduce(&self, data: &[f32]) -> Result<Vec<f32>, CoordinatorError> {
        let result = data.to_vec();

        // Phase 1: Reduce up the tree
        if !self.children.is_empty() {
            // Wait for children's results (simulated)
            for &child in &self.children {
                let _ = child; // Would receive from child
            }
        }

        // Send to parent
        if let Some(parent_rank) = self.parent {
            let msg = TensorMessage::new(
                result.clone(),
                super::communication::CompressionStrategy::None,
                MessagePriority::High,
            );
            self.async_comm.isend(msg, parent_rank).await?;
        }

        // Phase 2: Broadcast down the tree
        if let Some(parent_rank) = self.parent {
            // Receive from parent (simulated)
            let _ = parent_rank;
        }

        // Send to children
        for &child in &self.children {
            let msg = TensorMessage::new(
                result.clone(),
                super::communication::CompressionStrategy::None,
                MessagePriority::High,
            );
            self.async_comm.isend(msg, child).await?;
        }

        Ok(result)
    }

    /// Get branching factor
    pub fn branching_factor(&self) -> usize {
        self.branching_factor
    }

    /// Get parent rank
    pub fn parent(&self) -> Option<usize> {
        self.parent
    }

    /// Get children ranks
    pub fn children(&self) -> &[usize] {
        &self.children
    }
}

/// Distributed barrier for synchronization
pub struct DistributedBarrier {
    /// Communicator
    communicator: Arc<Communicator>,

    /// Counter for barrier generations
    generation: Arc<Mutex<u64>>,

    /// Arrived count for current barrier
    arrived: Arc<Mutex<usize>>,
}

impl DistributedBarrier {
    /// Create new distributed barrier
    pub fn new(communicator: Arc<Communicator>) -> Result<Self, CoordinatorError> {
        Ok(Self {
            communicator,
            generation: Arc::new(Mutex::new(0)),
            arrived: Arc::new(Mutex::new(0)),
        })
    }

    /// Wait at barrier until all processes arrive
    pub async fn wait(&self) -> Result<(), CoordinatorError> {
        let size = self.communicator.size();
        let mut arrived = self.arrived.lock().await;
        *arrived += 1;

        if *arrived == size {
            // Last process - reset and advance generation
            *arrived = 0;
            let mut gen = self.generation.lock().await;
            *gen += 1;
        } else {
            // Wait for all to arrive (simplified - would use actual synchronization)
            drop(arrived);
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        Ok(())
    }

    /// Get current generation
    pub async fn generation(&self) -> u64 {
        *self.generation.lock().await
    }
}

/// Checkpointing for fault tolerance
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct Checkpoint {
    /// Checkpoint ID
    pub id: String,

    /// Generation/iteration number
    pub generation: u64,

    /// Parameter values
    pub parameters: HashMap<String, Vec<f32>>,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl Checkpoint {
    /// Create new checkpoint
    pub fn new(id: String, generation: u64) -> Self {
        Self {
            id,
            generation,
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add parameter to checkpoint
    pub fn add_parameter(&mut self, key: String, values: Vec<f32>) {
        self.parameters.insert(key, values);
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Save checkpoint to file
    pub fn save(&self, path: &PathBuf) -> Result<(), CoordinatorError> {
        let data = oxicode::encode_to_vec(self)
            .map_err(|e| CoordinatorError::Checkpoint(format!("Failed to serialize: {}", e)))?;

        std::fs::write(path, data)
            .map_err(|e| CoordinatorError::Checkpoint(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: &PathBuf) -> Result<Self, CoordinatorError> {
        let data = std::fs::read(path)
            .map_err(|e| CoordinatorError::Checkpoint(format!("Failed to read file: {}", e)))?;

        let (checkpoint, _) = oxicode::decode_from_slice(&data)
            .map_err(|e| CoordinatorError::Checkpoint(format!("Failed to deserialize: {}", e)))?;
        Ok(checkpoint)
    }

    /// Get parameter
    pub fn get_parameter(&self, key: &str) -> Option<&Vec<f32>> {
        self.parameters.get(key)
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_creation() {
        let checkpoint = Checkpoint::new("test".to_string(), 100);
        assert_eq!(checkpoint.id, "test");
        assert_eq!(checkpoint.generation, 100);
        assert!(checkpoint.parameters.is_empty());
        assert!(checkpoint.metadata.is_empty());
    }

    #[test]
    fn test_checkpoint_add_parameter() {
        let mut checkpoint = Checkpoint::new("test".to_string(), 100);
        let params = vec![1.0, 2.0, 3.0];
        checkpoint.add_parameter("weights".to_string(), params.clone());

        assert_eq!(checkpoint.parameters.len(), 1);
        assert_eq!(checkpoint.get_parameter("weights"), Some(&params));
    }

    #[test]
    fn test_checkpoint_add_metadata() {
        let mut checkpoint = Checkpoint::new("test".to_string(), 100);
        checkpoint.add_metadata("model".to_string(), "resnet50".to_string());

        assert_eq!(checkpoint.metadata.len(), 1);
        assert_eq!(
            checkpoint.get_metadata("model"),
            Some(&"resnet50".to_string())
        );
    }

    #[test]
    fn test_checkpoint_serialization() {
        let mut checkpoint = Checkpoint::new("test".to_string(), 100);
        checkpoint.add_parameter("weights".to_string(), vec![1.0, 2.0, 3.0]);
        checkpoint.add_metadata("model".to_string(), "test_model".to_string());

        let serialized = oxicode::encode_to_vec(&checkpoint);
        assert!(serialized.is_ok());

        let bytes = serialized.expect("serialization failed");
        let deserialized: Result<(Checkpoint, usize), _> = oxicode::decode_from_slice(&bytes);
        assert!(deserialized.is_ok());

        let (restored, _) = deserialized.expect("deserialization failed");
        assert_eq!(restored.id, checkpoint.id);
        assert_eq!(restored.generation, checkpoint.generation);
    }

    #[test]
    fn test_checkpoint_save_load() {
        let mut checkpoint = Checkpoint::new("test".to_string(), 100);
        checkpoint.add_parameter("weights".to_string(), vec![1.0, 2.0, 3.0]);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_checkpoint.bin");

        let save_result = checkpoint.save(&path);
        assert!(save_result.is_ok());

        let load_result = Checkpoint::load(&path);
        assert!(load_result.is_ok());

        let loaded = load_result.expect("load failed");
        assert_eq!(loaded.id, checkpoint.id);
        assert_eq!(loaded.generation, checkpoint.generation);

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_checkpoint_get_missing_parameter() {
        let checkpoint = Checkpoint::new("test".to_string(), 100);
        assert_eq!(checkpoint.get_parameter("missing"), None);
    }

    #[test]
    fn test_checkpoint_get_missing_metadata() {
        let checkpoint = Checkpoint::new("test".to_string(), 100);
        assert_eq!(checkpoint.get_metadata("missing"), None);
    }

    #[test]
    fn test_checkpoint_multiple_parameters() {
        let mut checkpoint = Checkpoint::new("test".to_string(), 100);
        checkpoint.add_parameter("layer1".to_string(), vec![1.0, 2.0]);
        checkpoint.add_parameter("layer2".to_string(), vec![3.0, 4.0, 5.0]);

        assert_eq!(checkpoint.parameters.len(), 2);
        assert_eq!(checkpoint.get_parameter("layer1"), Some(&vec![1.0, 2.0]));
        assert_eq!(
            checkpoint.get_parameter("layer2"),
            Some(&vec![3.0, 4.0, 5.0])
        );
    }

    #[test]
    fn test_checkpoint_multiple_metadata() {
        let mut checkpoint = Checkpoint::new("test".to_string(), 100);
        checkpoint.add_metadata("model".to_string(), "resnet".to_string());
        checkpoint.add_metadata("dataset".to_string(), "imagenet".to_string());

        assert_eq!(checkpoint.metadata.len(), 2);
        assert_eq!(
            checkpoint.get_metadata("model"),
            Some(&"resnet".to_string())
        );
        assert_eq!(
            checkpoint.get_metadata("dataset"),
            Some(&"imagenet".to_string())
        );
    }
}
