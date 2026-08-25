//! Training Coordinators for Distributed Machine Learning
//!
//! This module provides coordination mechanisms for distributed training including
//! parameter servers, all-reduce implementations, barriers, and fault tolerance.
//!
//! # Features
//!
//! - **Parameter Server**: In-process parameter storage and updates —
//!   see [`ParameterServer`]'s docs for its single-rank restriction; it is
//!   *not* a networked multi-rank parameter server today.
//! - **All-Reduce**: Ring-AllReduce and tree-based implementations, both
//!   real multi-rank collectives over [`super::net::Endpoint`].
//! - **Barriers**: Synchronization primitives for coordinated execution
//! - **Fault Tolerance**: Checkpointing and recovery mechanisms
//! - **Load Balancing**: Dynamic work distribution
//!
//! # Architectures
//!
//! ## Parameter Server
//!
//! [`ParameterServer::new`] refuses any `communicator` with more than one
//! rank — see its docs. The diagram below is the *shape* the type's API
//! models (multiple logical workers pushing to, and pulling from, a
//! logical parameter server), which is genuine and correct as long as `W0`
//! and `PS0`/`PS1` are logical roles inside one process; nothing here
//! sends a push or a pull across a real rank boundary yet.
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
//! // Parameter server mode. `num_ps` (here 2) is a logical partitioning of
//! // keys across servers via `get_server_for_key`, *within* this one
//! // process — `ParameterServer::new` refuses a `world` with more than one
//! // rank (see its docs on why a real multi-rank parameter server needs
//! // routing this type does not yet provide).
//! let ps = ParameterServer::new(Arc::new(world.clone()), 2)?;
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

use super::collective::{self, allreduce, broadcast, reduce, CollectiveError, ReduceOp};
use super::communication::CommunicationError;
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

    /// A failure from a real [`super::collective`] operation — surfaced by
    /// [`RingAllReduce`]/[`TreeAllReduce`]/[`DistributedBarrier`], all of
    /// which now delegate to the real point-to-point transport rather than
    /// faking a local result.
    #[error("Collective operation error: {0}")]
    Collective(#[from] CollectiveError),

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

/// Parameter server for centralized parameter management.
///
/// # Single-process only
///
/// [`Self::parameters`]/[`Self::gradient_buffer`]/[`Self::versions`] are
/// plain in-memory maps behind a `tokio::sync` lock — process-local state,
/// never sent over `communicator`. That makes every method here correct
/// *within one process* (e.g. several logical workers sharing one
/// `ParameterServer` value on one rank) but silently wrong across ranks: a
/// real parameter-server deployment needs [`Self::push_gradients`] on
/// worker rank W to make its gradient visible to a `pull`/`apply` on a
/// different parameter-server rank P, and nothing here crosses that
/// boundary — each rank would quietly accumulate and apply only its own
/// pushes, with no error to say so.
///
/// [`Self::new`] therefore refuses any `communicator` with more than one
/// rank, rather than accepting one and *silently* running every rank as an
/// island. This is a capability gap, not a design choice this type
/// endorses: a real distributed parameter server needs point-to-point
/// routing keyed by [`Self::get_server_for_key`] (worker → owning PS rank)
/// built on the same real transport [`super::linalg`] and
/// [`super::collective`] already use — nobody has built that yet. Until
/// then, multi-rank callers should use [`RingAllReduce`]/[`TreeAllReduce`]
/// (both real, both delegate to [`super::collective::allreduce`]) instead.
pub struct ParameterServer {
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
    /// Create new parameter server.
    ///
    /// # Errors
    ///
    /// [`CoordinatorError::Configuration`] if `communicator` has more than
    /// one rank — see the type's docs on why a multi-rank `ParameterServer`
    /// would silently run every rank as an unsynchronized island rather
    /// than actually coordinating them.
    pub fn new(communicator: Arc<Communicator>, num_ps: usize) -> Result<Self, CoordinatorError> {
        if communicator.size() > 1 {
            return Err(CoordinatorError::Configuration(format!(
                "ParameterServer only coordinates within a single process (communicator has \
                 {} ranks); its parameter/gradient/version maps are process-local state that \
                 never crosses the network, so a multi-rank communicator here would silently run \
                 every rank as its own unsynchronized island rather than coordinating them. Use \
                 RingAllReduce or TreeAllReduce for real multi-rank gradient aggregation instead.",
                communicator.size()
            )));
        }

        Ok(Self {
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

/// Ring-AllReduce implementation for efficient gradient aggregation.
///
/// [`Self::allreduce`] delegates to [`super::collective::allreduce`] over
/// `communicator`'s real, shared [`super::net::endpoint::Endpoint`] — the
/// same reduce-then-broadcast (or, for large-enough payloads, ring
/// reduce-scatter+allgather) implementation every other collective caller in
/// this crate uses. This type's own name ("ring") no longer dictates the
/// wire algorithm actually run for a given call: `collective::allreduce`
/// picks the bandwidth-appropriate one internally (see its docs), and using
/// it here — rather than a bespoke, only-sometimes-correct reimplementation
/// of one specific algorithm — is what makes every call size- and
/// shape-correct rather than only correct in the cases a hand-rolled ring
/// loop happened to handle.
pub struct RingAllReduce {
    /// Communicator
    communicator: Arc<Communicator>,

    /// Ring topology (rank -> next rank), retained for [`Self::topology`]
    /// (kept as public API; not used by [`Self::allreduce`] itself, which
    /// goes through [`super::collective::allreduce`] instead of walking this
    /// ring by hand).
    ring: Vec<usize>,
}

impl RingAllReduce {
    /// Create new ring-allreduce coordinator
    pub fn new(communicator: Arc<Communicator>) -> Result<Self, CoordinatorError> {
        let size = communicator.size();

        // Build ring topology: 0 -> 1 -> 2 -> ... -> size-1 -> 0
        let ring: Vec<usize> = (0..size).map(|i| (i + 1) % size).collect();

        Ok(Self { communicator, ring })
    }

    /// Perform a real all-reduce (sum) on `data` across every rank in
    /// [`Self`]'s communicator, via [`super::collective::allreduce`].
    pub async fn allreduce(&self, data: &[f32]) -> Result<Vec<f32>, CoordinatorError> {
        Ok(allreduce(data, ReduceOp::Sum, &self.communicator).await?)
    }

    /// Get ring topology
    pub fn topology(&self) -> &[usize] {
        &self.ring
    }
}

/// Tree-based AllReduce for hierarchical communication.
///
/// [`Self::allreduce`] delegates to [`super::collective::allreduce`], the
/// same way [`RingAllReduce::allreduce`] does — see that type's docs for why.
/// `branching_factor`/[`Self::parent`]/[`Self::children`] are retained as
/// public API describing *a* tree shape consistent with `branching_factor`
/// (unchanged from before), but no longer drive the wire algorithm: this
/// type name's "tree" is a caller-facing label, not, as of this rewrite, a
/// distinct wire protocol from [`RingAllReduce`] — both now correctly
/// compute the same all-reduce result via the one real, already-tested
/// collective implementation, rather than each independently reimplementing
/// (and, before this rewrite, only partially implementing) their own
/// hand-rolled communication tree.
pub struct TreeAllReduce {
    /// Communicator
    communicator: Arc<Communicator>,

    /// Branching factor
    branching_factor: usize,

    /// Parent in tree (`None` for root)
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
            branching_factor,
            parent,
            children,
        })
    }

    /// Perform a real all-reduce (sum) on `data` across every rank in
    /// [`Self`]'s communicator, via [`super::collective::allreduce`].
    pub async fn allreduce(&self, data: &[f32]) -> Result<Vec<f32>, CoordinatorError> {
        Ok(allreduce(data, ReduceOp::Sum, &self.communicator).await?)
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

/// Distributed barrier for synchronization.
///
/// [`Self::wait`] delegates to [`super::process::Communicator::barrier`]'s
/// real dissemination algorithm (`ceil(log2(size))` rounds of real network
/// message exchange with peers `2^k` ahead/behind — see its docs), which
/// actually blocks every rank until every other rank has arrived. An earlier
/// version of this type instead incremented a private, per-process
/// `Arc<Mutex<usize>>` counter and fell back to a fixed `sleep(10ms)` for
/// every non-last caller: since each rank in a real distributed run is a
/// separate OS process (not merely a separate task in one process) with its
/// own independent memory, that counter could never actually observe another
/// rank's arrival at all — every multi-process call to it returned after
/// 10ms whether or not any other rank had actually reached the barrier.
/// [`Self::generation`] remains a local counter, but is now only ever
/// incremented immediately after a real barrier the whole communicator
/// participated in, so it genuinely counts completed barrier rounds.
pub struct DistributedBarrier {
    /// Communicator
    communicator: Arc<Communicator>,

    /// Counter for barrier generations: incremented after each real barrier
    /// this rank has completed (see the struct docs — this is bookkeeping
    /// derived from real synchronization, not a stand-in for it).
    generation: Arc<Mutex<u64>>,
}

impl DistributedBarrier {
    /// Create new distributed barrier
    pub fn new(communicator: Arc<Communicator>) -> Result<Self, CoordinatorError> {
        Ok(Self {
            communicator,
            generation: Arc::new(Mutex::new(0)),
        })
    }

    /// Wait at barrier until all processes arrive — a real network barrier
    /// via [`super::process::Communicator::barrier`], not a fabricated local
    /// wait (see the struct docs).
    pub async fn wait(&self) -> Result<(), CoordinatorError> {
        self.communicator.barrier().await?;
        let mut gen = self.generation.lock().await;
        *gen += 1;
        Ok(())
    }

    /// Get current generation: the number of real barriers this rank has
    /// completed via [`Self::wait`].
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

    /// [`ParameterServer::new`] must refuse a multi-rank communicator
    /// outright: its parameter/gradient/version maps are process-local and
    /// never cross the network, so silently accepting a multi-rank
    /// communicator would let every rank run as its own unsynchronized
    /// island with no error at all. This never needs to touch a real
    /// endpoint (the check runs before anything else), so an offline
    /// [`Communicator::new`] with `size > 1` is enough to exercise it.
    #[test]
    fn parameter_server_rejects_a_multi_rank_communicator() {
        let comm = Communicator::new(
            ProcessInfoForTest::info(0, 4),
            ProcessGroup::new(vec![0, 1, 2, 3]).expect("valid group"),
            HashMap::new(),
        )
        .expect("offline communicator");

        // `expect_err` needs `ParameterServer: Debug` (the success case),
        // which this type deliberately does not derive — match instead.
        match ParameterServer::new(Arc::new(comm), 2) {
            Err(CoordinatorError::Configuration(_)) => {}
            Err(other) => panic!("expected CoordinatorError::Configuration, got {other}"),
            Ok(_) => panic!("a 4-rank communicator must be rejected"),
        }
    }

    /// The single-rank case `ParameterServer::new` exists to allow: a
    /// world of exactly one rank, which cannot desynchronize from itself.
    #[test]
    fn parameter_server_accepts_a_single_rank_communicator() {
        let comm = Communicator::new(
            ProcessInfoForTest::info(0, 1),
            ProcessGroup::new(vec![0]).expect("valid group"),
            HashMap::new(),
        )
        .expect("offline communicator");

        assert!(ParameterServer::new(Arc::new(comm), 1).is_ok());
    }

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

    // -----------------------------------------------------------------
    // RingAllReduce / TreeAllReduce / DistributedBarrier over a real
    // LocalCluster — these now delegate to the real point-to-point
    // transport (super::collective / Communicator::barrier) instead of
    // fabricating a local result, so these tests exercise the genuine
    // network path.
    // -----------------------------------------------------------------

    use crate::distributed::process::{Communicator, ProcessGroup};
    use crate::distributed::testing::{ClusterNode, LocalCluster};
    use std::time::Duration;

    fn short_timeout_config() -> super::super::net::EndpointConfig {
        super::super::net::EndpointConfig {
            recv_timeout: Duration::from_secs(5),
            ..super::super::net::EndpointConfig::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ring_all_reduce_sums_across_a_real_cluster() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            4,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let reducer = RingAllReduce::new(Arc::new(comm.clone()))
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                let local = vec![(comm.rank() + 1) as f32; 6];
                let summed = reducer
                    .allreduce(&local)
                    .await
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                Ok(summed)
            },
        )
        .await
        .expect("ring all-reduce run");

        // Ranks 0..4 contribute (rank+1) each: 1+2+3+4 = 10, broadcast to all.
        for got in results {
            assert_eq!(got, vec![10.0; 6]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ring_all_reduce_topology_is_a_full_cycle() {
        let comm = Communicator::new(
            ProcessInfoForTest::info(0, 4),
            ProcessGroup::new(vec![0, 1, 2, 3]).expect("valid group"),
            HashMap::new(),
        )
        .expect("offline communicator");
        // topology() is unused by allreduce() itself now (see the type's
        // docs) but remains public API — pin it stays the same ring shape.
        let reducer = RingAllReduce::new(Arc::new(comm)).expect("reducer");
        assert_eq!(reducer.topology(), &[1, 2, 3, 0]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tree_all_reduce_sums_across_a_real_cluster() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            4,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let reducer = TreeAllReduce::new(Arc::new(comm.clone()), 2)
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                let local = vec![(comm.rank() + 1) as f32; 4];
                let summed = reducer
                    .allreduce(&local)
                    .await
                    .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                Ok(summed)
            },
        )
        .await
        .expect("tree all-reduce run");

        for got in results {
            assert_eq!(got, vec![10.0; 4]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tree_all_reduce_topology_matches_branching_factor() {
        let comm = Communicator::new(
            ProcessInfoForTest::info(3, 7),
            ProcessGroup::new((0..7).collect()).expect("valid group"),
            HashMap::new(),
        )
        .expect("offline communicator");
        let reducer = TreeAllReduce::new(Arc::new(comm), 2).expect("reducer");
        assert_eq!(reducer.branching_factor(), 2);
        // rank 3: parent = (3-1)/2 = 1; children = 3*2+1=7 (out of range for
        // size 7), 3*2+2=8 (out of range) -> no children.
        assert_eq!(reducer.parent(), Some(1));
        assert!(reducer.children().is_empty());
    }

    /// Every rank stamps a shared, process-local phase counter immediately
    /// before calling `wait`, staggered so ranks arrive at very different
    /// times; if any rank could observe the counter right after its own
    /// `wait` returns without every rank having stamped it first, that would
    /// prove `wait` is *not* a real cross-process barrier — the same
    /// regression shape as `process::tests::barrier_releases_all_ranks_together`,
    /// now specifically pinned against `DistributedBarrier` (whose `wait`
    /// used to fabricate a fixed 10ms sleep instead of really synchronizing).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distributed_barrier_wait_releases_all_ranks_together() {
        let cfg = short_timeout_config();
        let phase = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let phase_for_body = Arc::clone(&phase);
        let results = LocalCluster::run_connected_with(
            4,
            cfg,
            Duration::from_secs(15),
            move |node: ClusterNode| {
                let phase = Arc::clone(&phase_for_body);
                async move {
                    let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                    let barrier = DistributedBarrier::new(Arc::new(comm.clone()))
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    tokio::time::sleep(Duration::from_millis(5 * comm.rank() as u64)).await;
                    phase.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    barrier
                        .wait()
                        .await
                        .map_err(|e| super::super::net::NetError::Io(e.to_string()))?;
                    let seen = phase.load(std::sync::atomic::Ordering::SeqCst);
                    let generation = barrier.generation().await;
                    Ok((seen, generation))
                }
            },
        )
        .await
        .expect("barrier run");

        for (seen, generation) in results {
            assert_eq!(
                seen, 4,
                "a rank exited DistributedBarrier::wait before every rank had arrived"
            );
            assert_eq!(
                generation, 1,
                "one real wait() must advance generation by exactly 1"
            );
        }
    }

    #[tokio::test]
    async fn distributed_barrier_generation_advances_once_per_wait_at_size_one() {
        let comm = Communicator::new(
            ProcessInfoForTest::info(0, 1),
            ProcessGroup::new(vec![0]).expect("valid group"),
            HashMap::new(),
        )
        .expect("offline communicator");
        let barrier = DistributedBarrier::new(Arc::new(comm)).expect("barrier");
        assert_eq!(barrier.generation().await, 0);
        barrier
            .wait()
            .await
            .expect("size-1 barrier is a real no-op");
        assert_eq!(barrier.generation().await, 1);
        barrier
            .wait()
            .await
            .expect("size-1 barrier is a real no-op");
        assert_eq!(barrier.generation().await, 2);
    }

    /// Small helper so the offline-communicator tests above don't need to
    /// repeat `ProcessInfo::new(...).expect(...)` with a throwaway address
    /// and hostname each time.
    struct ProcessInfoForTest;
    impl ProcessInfoForTest {
        fn info(rank: usize, size: usize) -> super::super::process::ProcessInfo {
            let addr: std::net::SocketAddr = "127.0.0.1:5000".parse().expect("valid address");
            super::super::process::ProcessInfo::new(rank, size, addr, "localhost".to_string())
                .expect("valid process info")
        }
    }
}
