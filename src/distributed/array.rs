//! Distributed Array Structures and Partitioning Strategies
//!
//! This module provides distributed array data structures that automatically partition
//! data across multiple processes for parallel computation.
//!
//! # Partitioning Strategies
//!
//! - **Block**: Contiguous chunks (good for locality)
//! - **Cyclic**: Round-robin distribution (good for load balancing)
//! - **Block-Cyclic**: Hybrid approach combining both
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::array::*;
//! use numrs2::distributed::process::*;
//!
//! # async fn example() -> Result<(), DistributedArrayError> {
//! let world = init().await?;
//!
//! // Create distributed array with block distribution
//! let global_size = 1000;
//! let local_data: Vec<f64> = vec![world.rank() as f64; global_size / world.size()];
//! let dist_array = DistributedArray::from_local(
//!     local_data,
//!     DistributionStrategy::Block,
//!     global_size,
//!     &world
//! )?;
//!
//! // Access local portion
//! let local = dist_array.local_data();
//! println!("Local data size: {}", local.len());
//!
//! // Convert between global and local indices
//! let global_idx = GlobalIndex::new(500);
//! if let Some(local_idx) = dist_array.global_to_local(&global_idx)? {
//!     println!("Global 500 is local {}", local_idx.index());
//! }
//!
//! finalize(world).await?;
//! # Ok(())
//! # }
//! ```

use super::collective::{
    allgather, allreduce, gather, recv_vec, scatter, send_slice, CollectiveError, ReduceOp,
};
use super::process::{Communicator, ProcessError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ===========================================================================
// Wire tags for ghost-cell boundary exchange
//
// This is point-to-point, not a collective, but it shares one communicator's
// `ctx` (always `comm.context().0`) with every collective in
// `super::collective`, so it needs its own tag range those can never step
// into. `super::collective`'s own `TAG_*_BASE` constants occupy
// `0x1_0000_0000` through `0xA_0000_0000` (one `0x1_0000_0000`-wide band per
// collective, each call's rounds folded into the low bits of its band — see
// that module's tag-block comment), so `0x10_0000_0000` and up is
// unreachable to any of them regardless of world size.
// ===========================================================================
const TAG_GHOST_TO_RIGHT: u64 = 0x10_0000_0000;
const TAG_GHOST_TO_LEFT: u64 = 0x11_0000_0000;

/// Errors that can occur with distributed arrays
#[derive(Error, Debug)]
pub enum DistributedArrayError {
    #[error("Collective operation error: {0}")]
    Collective(#[from] CollectiveError),

    #[error("Process error: {0}")]
    Process(#[from] ProcessError),

    #[error("Invalid global index {index}, array size is {size}")]
    InvalidGlobalIndex { index: usize, size: usize },

    #[error("Invalid local index {index}, local size is {size}")]
    InvalidLocalIndex { index: usize, size: usize },

    #[error("Size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("Distribution error: {0}")]
    DistributionError(String),

    #[error("Ghost cell error: {0}")]
    GhostCellError(String),

    #[error("Partitioning error: {0}")]
    PartitionError(String),
}

/// Distribution strategy for partitioning arrays across processes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionStrategy {
    /// Block distribution: contiguous chunks
    /// Process 0: [0..n/p), Process 1: [n/p..2n/p), etc.
    Block,

    /// Cyclic distribution: round-robin
    /// Process 0: [0, p, 2p, ...], Process 1: [1, p+1, 2p+1, ...], etc.
    Cyclic,

    /// Block-cyclic distribution with specified block size
    /// Combines block and cyclic: blocks of size k distributed cyclically
    BlockCyclic { block_size: usize },
}

impl DistributionStrategy {
    /// Calculate the owner (process rank) of a global index
    pub fn owner(&self, global_idx: usize, global_size: usize, num_processes: usize) -> usize {
        match self {
            DistributionStrategy::Block => {
                // Block distribution: divide array into equal-sized chunks
                let base_size = global_size / num_processes;
                let remainder = global_size % num_processes;

                // Calculate which process owns this index
                if global_idx < remainder * (base_size + 1) {
                    // In the region where processes have base_size + 1 elements
                    global_idx / (base_size + 1)
                } else {
                    // In the region where processes have base_size elements
                    let offset = remainder * (base_size + 1);
                    remainder + (global_idx - offset) / base_size
                }
            }
            DistributionStrategy::Cyclic => {
                // Cyclic distribution: round-robin
                global_idx % num_processes
            }
            DistributionStrategy::BlockCyclic { block_size } => {
                // Block-cyclic: blocks distributed cyclically
                (global_idx / block_size) % num_processes
            }
        }
    }

    /// Calculate local size for a process
    pub fn local_size(&self, global_size: usize, rank: usize, num_processes: usize) -> usize {
        match self {
            DistributionStrategy::Block => {
                let base_size = global_size / num_processes;
                let remainder = global_size % num_processes;
                if rank < remainder {
                    base_size + 1
                } else {
                    base_size
                }
            }
            DistributionStrategy::Cyclic => {
                (global_size + num_processes - 1 - rank) / num_processes
            }
            DistributionStrategy::BlockCyclic { block_size } => {
                let num_blocks = global_size.div_ceil(*block_size);
                let blocks_per_proc = num_blocks / num_processes;
                let extra_blocks = num_blocks % num_processes;

                let my_blocks = if rank < extra_blocks {
                    blocks_per_proc + 1
                } else {
                    blocks_per_proc
                };

                // Last block might be partial
                let last_block_start = (num_blocks - 1) * block_size;
                let last_block_owner = (num_blocks - 1) % num_processes;

                if rank == last_block_owner {
                    (my_blocks - 1) * block_size + (global_size - last_block_start)
                } else {
                    my_blocks * block_size
                }
            }
        }
    }
}

/// Global index in the distributed array
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GlobalIndex(usize);

impl GlobalIndex {
    /// Create a new global index
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Get the index value
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Local index within a process's portion of the array
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LocalIndex(usize);

impl LocalIndex {
    /// Create a new local index
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Get the index value
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Ghost cells for boundary data synchronization
#[derive(Debug, Clone)]
pub struct GhostCells<T> {
    /// Left ghost cells (from previous rank)
    left: Vec<T>,
    /// Right ghost cells (from next rank)
    right: Vec<T>,
    /// Number of ghost cells on each side
    width: usize,
}

impl<T: Clone> GhostCells<T> {
    /// Create new ghost cells with specified width
    pub fn new(width: usize) -> Self {
        Self {
            left: Vec::with_capacity(width),
            right: Vec::with_capacity(width),
            width,
        }
    }

    /// Get left ghost cells
    pub fn left(&self) -> &[T] {
        &self.left
    }

    /// Get right ghost cells
    pub fn right(&self) -> &[T] {
        &self.right
    }

    /// Get ghost cell width
    pub fn width(&self) -> usize {
        self.width
    }

    /// Set left ghost cells
    pub fn set_left(&mut self, data: Vec<T>) {
        self.left = data;
    }

    /// Set right ghost cells
    pub fn set_right(&mut self, data: Vec<T>) {
        self.right = data;
    }
}

/// Distributed array with automatic partitioning
pub struct DistributedArray<T> {
    /// Local portion of the array
    local_data: Vec<T>,
    /// Global size of the array
    global_size: usize,
    /// Distribution strategy
    strategy: DistributionStrategy,
    /// Communicator for this array
    comm: Communicator,
    /// Ghost cells for boundary synchronization
    ghost_cells: Option<GhostCells<T>>,
}

impl<T: Clone + Serialize + for<'de> Deserialize<'de> + Send + 'static> DistributedArray<T> {
    /// Create a distributed array from local data
    pub fn from_local(
        local_data: Vec<T>,
        strategy: DistributionStrategy,
        global_size: usize,
        comm: &Communicator,
    ) -> Result<Self, DistributedArrayError> {
        // Validate local size matches expected size for this rank
        let expected_size = strategy.local_size(global_size, comm.rank(), comm.size());
        if local_data.len() != expected_size {
            return Err(DistributedArrayError::SizeMismatch {
                expected: expected_size,
                actual: local_data.len(),
            });
        }

        Ok(Self {
            local_data,
            global_size,
            strategy,
            comm: comm.clone(),
            ghost_cells: None,
        })
    }

    /// Create a distributed array by scattering from root
    pub async fn scatter_from_root(
        data: Vec<T>,
        strategy: DistributionStrategy,
        root: usize,
        comm: &Communicator,
    ) -> Result<Self, DistributedArrayError> {
        let global_size = if comm.rank() == root { data.len() } else { 0 };

        // Scatter data to all processes
        let local_data = scatter(&data, root, comm).await?;

        Ok(Self {
            local_data,
            global_size,
            strategy,
            comm: comm.clone(),
            ghost_cells: None,
        })
    }

    /// Gather distributed array at root
    pub async fn gather_at_root(&self, root: usize) -> Result<Vec<T>, DistributedArrayError> {
        let gathered = gather(&self.local_data, root, &self.comm).await?;
        Ok(gathered)
    }

    /// Gather distributed array at all processes
    pub async fn allgather(&self) -> Result<Vec<T>, DistributedArrayError> {
        let gathered = allgather(&self.local_data, &self.comm).await?;
        Ok(gathered)
    }

    /// Get local data
    pub fn local_data(&self) -> &[T] {
        &self.local_data
    }

    /// Get mutable local data
    pub fn local_data_mut(&mut self) -> &mut [T] {
        &mut self.local_data
    }

    /// Get global size
    pub fn global_size(&self) -> usize {
        self.global_size
    }

    /// Get local size
    pub fn local_size(&self) -> usize {
        self.local_data.len()
    }

    /// Get distribution strategy
    pub fn strategy(&self) -> DistributionStrategy {
        self.strategy
    }

    /// Get communicator
    pub fn comm(&self) -> &Communicator {
        &self.comm
    }

    /// Convert global index to local index (if owned by this process)
    pub fn global_to_local(
        &self,
        global_idx: &GlobalIndex,
    ) -> Result<Option<LocalIndex>, DistributedArrayError> {
        let idx = global_idx.index();

        if idx >= self.global_size {
            return Err(DistributedArrayError::InvalidGlobalIndex {
                index: idx,
                size: self.global_size,
            });
        }

        let owner = self.strategy.owner(idx, self.global_size, self.comm.size());
        if owner != self.comm.rank() {
            return Ok(None);
        }

        // Calculate local index based on strategy
        let local_idx = match self.strategy {
            DistributionStrategy::Block => {
                let base_size = self.global_size / self.comm.size();
                let remainder = self.global_size % self.comm.size();
                let offset = if self.comm.rank() < remainder {
                    self.comm.rank() * (base_size + 1)
                } else {
                    remainder * (base_size + 1) + (self.comm.rank() - remainder) * base_size
                };
                idx - offset
            }
            DistributionStrategy::Cyclic => idx / self.comm.size(),
            DistributionStrategy::BlockCyclic { block_size } => {
                let block = idx / block_size;
                let offset_in_block = idx % block_size;
                (block / self.comm.size()) * block_size + offset_in_block
            }
        };

        Ok(Some(LocalIndex::new(local_idx)))
    }

    /// Convert local index to global index
    pub fn local_to_global(
        &self,
        local_idx: &LocalIndex,
    ) -> Result<GlobalIndex, DistributedArrayError> {
        let idx = local_idx.index();

        if idx >= self.local_data.len() {
            return Err(DistributedArrayError::InvalidLocalIndex {
                index: idx,
                size: self.local_data.len(),
            });
        }

        let global_idx = match self.strategy {
            DistributionStrategy::Block => {
                let base_size = self.global_size / self.comm.size();
                let remainder = self.global_size % self.comm.size();
                let offset = if self.comm.rank() < remainder {
                    self.comm.rank() * (base_size + 1)
                } else {
                    remainder * (base_size + 1) + (self.comm.rank() - remainder) * base_size
                };
                offset + idx
            }
            DistributionStrategy::Cyclic => idx * self.comm.size() + self.comm.rank(),
            DistributionStrategy::BlockCyclic { block_size } => {
                let block_number = idx / block_size;
                let offset_in_block = idx % block_size;
                (block_number * self.comm.size() + self.comm.rank()) * block_size + offset_in_block
            }
        };

        Ok(GlobalIndex::new(global_idx))
    }

    /// Initialize ghost cells with specified width
    pub fn init_ghost_cells(&mut self, width: usize) {
        self.ghost_cells = Some(GhostCells::new(width));
    }

    /// Synchronize ghost cells with neighboring processes.
    ///
    /// Real point-to-point boundary exchange over [`super::collective`]'s
    /// `Endpoint`-backed `send_slice`/`recv_vec` — an earlier version of this
    /// method dropped every boundary on the floor and always populated both
    /// ghost regions with an empty `vec![]`, regardless of what a neighbor
    /// actually held. Rank 0 has no left neighbor and the last rank has no
    /// right neighbor, so each half of the exchange is skipped there, and
    /// [`GhostCells::left`]/[`GhostCells::right`] stay at their
    /// [`GhostCells::new`] default (empty) on those ends.
    pub async fn sync_ghost_cells(&mut self) -> Result<(), DistributedArrayError> {
        let width = self
            .ghost_cells
            .as_ref()
            .ok_or_else(|| {
                DistributedArrayError::GhostCellError("Ghost cells not initialized".to_string())
            })?
            .width();

        // Without this check, a rank whose local block is narrower than
        // `width` would slice its *entire* block as the boundary (via
        // `saturating_sub`/a `min`-clamped range) and a neighbor would
        // silently store a short ghost region instead of the `width`
        // elements it asked for — reject the mismatch instead.
        //
        // This must be a *collective* decision, not a rank-local early
        // return: every rank below the threshold does return `Err` on its
        // own, correctly, but a rank *at or above* it would fall through
        // to the sends/receives below and then block forever on a peer
        // that took the error path and never sent anything. `allreduce`'s
        // `Max` here turns "does any rank's local block violate the
        // width?" into a value every rank agrees on before anyone commits
        // to a send.
        let violation = if width > self.local_data.len() {
            1u32
        } else {
            0
        };
        let any_violation = allreduce(&[violation], ReduceOp::Max, &self.comm)
            .await
            .map_err(DistributedArrayError::from)?;
        if any_violation.first().copied().unwrap_or(0) > 0 {
            return Err(DistributedArrayError::GhostCellError(format!(
                "ghost cell width {width} exceeds some rank's local size (this rank: {})",
                self.local_data.len()
            )));
        }

        let rank = self.comm.rank();
        let size = self.comm.size();

        // Send right boundary to next rank, receive left boundary from previous rank.
        if rank < size - 1 {
            let right_boundary = self.local_data[self.local_data.len() - width..].to_vec();
            send_slice(&self.comm, rank + 1, TAG_GHOST_TO_RIGHT, &right_boundary).await?;
        }
        if rank > 0 {
            let left_boundary: Vec<T> = recv_vec(&self.comm, rank - 1, TAG_GHOST_TO_RIGHT).await?;
            self.ghost_cells
                .as_mut()
                .ok_or_else(|| {
                    DistributedArrayError::GhostCellError(
                        "ghost cells were cleared mid-sync".to_string(),
                    )
                })?
                .set_left(left_boundary);
        }

        // Send left boundary to previous rank, receive right boundary from next rank.
        if rank > 0 {
            let left_boundary = self.local_data[..width].to_vec();
            send_slice(&self.comm, rank - 1, TAG_GHOST_TO_LEFT, &left_boundary).await?;
        }
        if rank < size - 1 {
            let right_boundary: Vec<T> = recv_vec(&self.comm, rank + 1, TAG_GHOST_TO_LEFT).await?;
            self.ghost_cells
                .as_mut()
                .ok_or_else(|| {
                    DistributedArrayError::GhostCellError(
                        "ghost cells were cleared mid-sync".to_string(),
                    )
                })?
                .set_right(right_boundary);
        }

        Ok(())
    }

    /// Get ghost cells
    pub fn ghost_cells(&self) -> Option<&GhostCells<T>> {
        self.ghost_cells.as_ref()
    }
}

impl<T: Clone + std::fmt::Debug> std::fmt::Debug for DistributedArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedArray")
            .field("global_size", &self.global_size)
            .field("local_size", &self.local_data.len())
            .field("strategy", &self.strategy)
            .field("rank", &self.comm.rank())
            .field("size", &self.comm.size())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_strategy_block_owner() {
        let strategy = DistributionStrategy::Block;
        let global_size = 100;
        let num_processes = 4;

        // For a 100-element array with 4 processes:
        // Process 0: [0..25), Process 1: [25..50), Process 2: [50..75), Process 3: [75..100)
        assert_eq!(strategy.owner(0, global_size, num_processes), 0);
        assert_eq!(strategy.owner(24, global_size, num_processes), 0);
        assert_eq!(strategy.owner(25, global_size, num_processes), 1);
        assert_eq!(strategy.owner(74, global_size, num_processes), 2);
    }

    #[test]
    fn test_distribution_strategy_cyclic_owner() {
        let strategy = DistributionStrategy::Cyclic;
        let global_size = 100;
        let num_processes = 4;

        assert_eq!(strategy.owner(0, global_size, num_processes), 0);
        assert_eq!(strategy.owner(1, global_size, num_processes), 1);
        assert_eq!(strategy.owner(2, global_size, num_processes), 2);
        assert_eq!(strategy.owner(3, global_size, num_processes), 3);
        assert_eq!(strategy.owner(4, global_size, num_processes), 0);
        assert_eq!(strategy.owner(5, global_size, num_processes), 1);
    }

    #[test]
    fn test_distribution_strategy_block_local_size() {
        let strategy = DistributionStrategy::Block;
        let global_size = 100;
        let num_processes = 4;

        assert_eq!(strategy.local_size(global_size, 0, num_processes), 25);
        assert_eq!(strategy.local_size(global_size, 1, num_processes), 25);
        assert_eq!(strategy.local_size(global_size, 2, num_processes), 25);
        assert_eq!(strategy.local_size(global_size, 3, num_processes), 25);
    }

    #[test]
    fn test_distribution_strategy_block_local_size_uneven() {
        let strategy = DistributionStrategy::Block;
        let global_size = 103;
        let num_processes = 4;

        // First 3 processes get 26 elements, last gets 25
        assert_eq!(strategy.local_size(global_size, 0, num_processes), 26);
        assert_eq!(strategy.local_size(global_size, 1, num_processes), 26);
        assert_eq!(strategy.local_size(global_size, 2, num_processes), 26);
        assert_eq!(strategy.local_size(global_size, 3, num_processes), 25);
    }

    #[test]
    fn test_global_index() {
        let idx = GlobalIndex::new(42);
        assert_eq!(idx.index(), 42);
    }

    #[test]
    fn test_local_index() {
        let idx = LocalIndex::new(10);
        assert_eq!(idx.index(), 10);
    }

    #[test]
    fn test_ghost_cells() {
        let mut ghost: GhostCells<f64> = GhostCells::new(3);
        assert_eq!(ghost.width(), 3);

        ghost.set_left(vec![1.0, 2.0, 3.0]);
        ghost.set_right(vec![4.0, 5.0, 6.0]);

        assert_eq!(ghost.left(), &[1.0, 2.0, 3.0]);
        assert_eq!(ghost.right(), &[4.0, 5.0, 6.0]);
    }

    // -----------------------------------------------------------------
    // sync_ghost_cells: real point-to-point exchange over a LocalCluster.
    // An earlier version never sent or received anything and always set
    // both ghost regions to an empty vec![], regardless of a neighbor's
    // actual data — these exercise the genuine network path instead.
    // -----------------------------------------------------------------

    use crate::distributed::net::NetError;
    use crate::distributed::process::{ProcessGroup, ProcessInfo};
    use crate::distributed::testing::{ClusterNode, LocalCluster};
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::sync::Arc;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sync_ghost_cells_exchanges_real_neighbor_boundaries() {
        const WORLD_SIZE: u32 = 4;
        const PER_RANK: usize = 5;
        const WIDTH: usize = 2;

        let results = LocalCluster::run_connected(WORLD_SIZE, |node: ClusterNode| async move {
            let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
            let rank = comm.rank();

            let local_data: Vec<f64> = (0..PER_RANK).map(|i| (rank * 100 + i) as f64).collect();
            let mut array = DistributedArray::from_local(
                local_data,
                DistributionStrategy::Block,
                PER_RANK * WORLD_SIZE as usize,
                &comm,
            )
            .map_err(|e| NetError::Io(e.to_string()))?;

            array.init_ghost_cells(WIDTH);
            array
                .sync_ghost_cells()
                .await
                .map_err(|e| NetError::Io(e.to_string()))?;

            let ghosts = array.ghost_cells().ok_or_else(|| {
                NetError::Io("ghost cells vanished after sync_ghost_cells".to_string())
            })?;
            Ok((ghosts.left().to_vec(), ghosts.right().to_vec()))
        })
        .await
        .expect("cluster run should succeed");

        // Rank 0 has no left neighbor: left ghost stays at its `new` default (empty).
        assert_eq!(results[0].0, Vec::<f64>::new());
        assert_eq!(
            results[0].1,
            vec![100.0, 101.0],
            "rank 0's right ghost is rank 1's first WIDTH elements"
        );

        assert_eq!(
            results[1].0,
            vec![3.0, 4.0],
            "rank 1's left ghost is rank 0's last WIDTH elements"
        );
        assert_eq!(results[1].1, vec![200.0, 201.0]);

        assert_eq!(results[2].0, vec![103.0, 104.0]);
        assert_eq!(results[2].1, vec![300.0, 301.0]);

        assert_eq!(results[3].0, vec![203.0, 204.0]);
        // The last rank has no right neighbor: right ghost stays empty.
        assert_eq!(results[3].1, Vec::<f64>::new());
    }

    #[tokio::test]
    async fn sync_ghost_cells_rejects_a_width_wider_than_the_local_block() {
        // A size-one offline `Communicator` needs no tokio-bound socket: both
        // neighbor branches are unreachable at world size 1, so the width
        // check is the only thing this test can be exercising.
        let addr: SocketAddr = "127.0.0.1:5000".parse().expect("valid addr");
        let info = ProcessInfo::new(0, 1, addr, "localhost".to_string()).expect("valid info");
        let group = ProcessGroup::new(vec![0]).expect("valid group");
        let comm = Communicator::new(info, group, HashMap::new()).expect("valid communicator");

        let mut array = DistributedArray::from_local(
            vec![1.0_f64, 2.0, 3.0],
            DistributionStrategy::Block,
            3,
            &comm,
        )
        .expect("valid distributed array");

        array.init_ghost_cells(5); // wider than the 3-element local block
        let err = array
            .sync_ghost_cells()
            .await
            .expect_err("width wider than the local block must be rejected");
        assert!(matches!(err, DistributedArrayError::GhostCellError(_)));
    }
}
