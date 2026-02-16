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

use super::collective::{allgather, gather, scatter, CollectiveError};
use super::process::{Communicator, ProcessError};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

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

    /// Synchronize ghost cells with neighboring processes
    pub async fn sync_ghost_cells(&mut self) -> Result<(), DistributedArrayError>
    where
        T: Clone,
    {
        let ghost_cells = self.ghost_cells.as_mut().ok_or_else(|| {
            DistributedArrayError::GhostCellError("Ghost cells not initialized".to_string())
        })?;

        let width = ghost_cells.width();
        let rank = self.comm.rank();
        let size = self.comm.size();

        // Send right boundary to next rank, receive left boundary from previous rank
        if rank < size - 1 {
            let right_boundary =
                self.local_data[self.local_data.len().saturating_sub(width)..].to_vec();
            // In real implementation: send right_boundary to rank + 1
            // For now, placeholder
            let _ = right_boundary;
        }

        if rank > 0 {
            // In real implementation: receive left boundary from rank - 1
            // For now, placeholder
            let left_boundary = vec![];
            ghost_cells.set_left(left_boundary);
        }

        // Send left boundary to previous rank, receive right boundary from next rank
        if rank > 0 {
            let left_boundary = self.local_data[..width.min(self.local_data.len())].to_vec();
            // In real implementation: send left_boundary to rank - 1
            let _ = left_boundary;
        }

        if rank < size - 1 {
            // In real implementation: receive right boundary from rank + 1
            let right_boundary = vec![];
            ghost_cells.set_right(right_boundary);
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
}
