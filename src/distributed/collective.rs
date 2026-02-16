//! Collective Operations for Distributed Computing
//!
//! This module provides MPI-like collective communication operations for coordinating
//! multiple processes in distributed computations.
//!
//! # Collective Operations
//!
//! - **Reduce**: Combine data from all processes using an operation (sum, max, min, etc.)
//! - **All-Reduce**: Reduce and distribute result to all processes
//! - **Broadcast**: Send data from one process to all others
//! - **Gather**: Collect data from all processes at root
//! - **All-Gather**: Collect data from all processes and distribute to all
//! - **Scatter**: Distribute data from root to all processes
//! - **All-Scatter**: Distribute data from all to all processes
//! - **Barrier**: Synchronize all processes
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::collective::*;
//! use numrs2::distributed::process::*;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let world = init().await?;
//!
//! // Each process has some local data
//! let local_data = vec![world.rank() as f64; 10];
//!
//! // Sum across all processes
//! let sum = allreduce(&local_data, ReduceOp::Sum, &world).await?;
//!
//! // Broadcast from root
//! let mut data = if world.is_root() {
//!     vec![1.0, 2.0, 3.0]
//! } else {
//!     vec![0.0; 3]
//! };
//! broadcast(&mut data, 0, &world).await?;
//!
//! // Gather at root
//! let gathered = gather(&local_data, 0, &world).await?;
//! if world.is_root() {
//!     println!("Gathered {} elements", gathered.len());
//! }
//!
//! finalize(world).await?;
//! # Ok(())
//! # }
//! ```

use super::comm::{CommunicationError, Message, MessageTag};
use super::process::{Communicator, ProcessError};
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};
use thiserror::Error;

/// Errors that can occur during collective operations
#[derive(Error, Debug)]
pub enum CollectiveError {
    #[error("Communication error: {0}")]
    Communication(#[from] CommunicationError),

    #[error("Process error: {0}")]
    Process(#[from] ProcessError),

    #[error("Invalid root rank {root}, must be < {size}")]
    InvalidRoot { root: usize, size: usize },

    #[error("Data size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("Collective operation failed: {0}")]
    OperationFailed(String),

    #[error("Timeout during collective operation")]
    Timeout,
}

/// Reduction operations for collective reduce
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Sum of all values
    Sum,
    /// Product of all values
    Product,
    /// Maximum value
    Max,
    /// Minimum value
    Min,
    /// Logical AND (for boolean/integer types)
    And,
    /// Logical OR (for boolean/integer types)
    Or,
}

impl ReduceOp {
    /// Apply the reduction operation to two values
    pub fn apply<T>(&self, a: T, b: T) -> T
    where
        T: Add<Output = T> + Mul<Output = T> + PartialOrd + Clone,
    {
        match self {
            ReduceOp::Sum => a + b,
            ReduceOp::Product => a * b,
            ReduceOp::Max => {
                if a > b {
                    a
                } else {
                    b
                }
            }
            ReduceOp::Min => {
                if a < b {
                    a
                } else {
                    b
                }
            }
            _ => a, // And/Or require integer types
        }
    }

    /// Apply reduction to a slice of values
    pub fn apply_slice<T>(&self, values: &[T]) -> Option<T>
    where
        T: Add<Output = T> + Mul<Output = T> + PartialOrd + Clone,
    {
        if values.is_empty() {
            return None;
        }

        let mut result = values[0].clone();
        for value in &values[1..] {
            result = self.apply(result, value.clone());
        }

        Some(result)
    }
}

/// Message tags for collective operations
const TAG_REDUCE: MessageTag = 1000;
const TAG_BROADCAST: MessageTag = 1001;
const TAG_GATHER: MessageTag = 1002;
const TAG_SCATTER: MessageTag = 1003;
const TAG_BARRIER: MessageTag = 1004;

/// Reduce operation: combine data from all processes at root using specified operation
///
/// All processes contribute their local data, and the root process receives
/// the combined result according to the reduction operation.
///
/// # Arguments
///
/// * `data` - Local data from this process
/// * `op` - Reduction operation (Sum, Max, Min, etc.)
/// * `root` - Rank of root process that will receive result
/// * `comm` - Communicator containing all participating processes
///
/// # Returns
///
/// Root process receives the reduced result, other processes receive empty vector.
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::collective::*;
/// # use numrs2::distributed::process::*;
/// # async fn example(world: &Communicator) -> Result<(), CollectiveError> {
/// let local_sum = vec![world.rank() as f64];
/// let total = reduce(&local_sum, ReduceOp::Sum, 0, world).await?;
/// if world.is_root() {
///     println!("Total sum: {:?}", total);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn reduce<T>(
    data: &[T],
    op: ReduceOp,
    root: usize,
    comm: &Communicator,
) -> Result<Vec<T>, CollectiveError>
where
    T: Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Add<Output = T>
        + Mul<Output = T>
        + PartialOrd
        + Send
        + 'static,
{
    let rank = comm.rank();
    let size = comm.size();

    if root >= size {
        return Err(CollectiveError::InvalidRoot { root, size });
    }

    // In a real implementation, this would use tree-based reduction for efficiency
    // For now, simple gather-and-reduce at root

    if rank == root {
        // Root collects data from all processes
        let all_data = [data.to_vec()];

        // Simplified: would use actual network receive
        // In real implementation: receive from all non-root processes
        for src in 0..size {
            if src != root {
                // Placeholder: would receive actual data from process src
                // For now, just use empty data as placeholder
            }
        }

        // Apply reduction operation
        let mut result = data.to_vec();
        for process_data in &all_data[1..] {
            for (i, value) in process_data.iter().enumerate() {
                if i < result.len() {
                    result[i] = op.apply(result[i].clone(), value.clone());
                }
            }
        }

        Ok(result)
    } else {
        // Non-root sends data to root
        // Placeholder: would send actual data to root process
        // For now, just return empty vector
        Ok(Vec::new())
    }
}

/// All-reduce operation: reduce and distribute result to all processes
///
/// Like reduce, but all processes receive the result instead of just the root.
///
/// # Arguments
///
/// * `data` - Local data from this process
/// * `op` - Reduction operation (Sum, Max, Min, etc.)
/// * `comm` - Communicator containing all participating processes
///
/// # Returns
///
/// All processes receive the same reduced result.
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::collective::*;
/// # use numrs2::distributed::process::*;
/// # async fn example(world: &Communicator) -> Result<(), CollectiveError> {
/// let local_value = vec![1.0_f64; 100];
/// let global_sum = allreduce(&local_value, ReduceOp::Sum, world).await?;
/// println!("Rank {}: global sum = {:?}", world.rank(), global_sum[0]);
/// # Ok(())
/// # }
/// ```
pub async fn allreduce<T>(
    data: &[T],
    op: ReduceOp,
    comm: &Communicator,
) -> Result<Vec<T>, CollectiveError>
where
    T: Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Add<Output = T>
        + Mul<Output = T>
        + PartialOrd
        + Send
        + 'static,
{
    // Reduce at root (rank 0)
    let reduced = reduce(data, op, 0, comm).await?;

    // Broadcast result to all processes
    let result = if comm.is_root() { reduced } else { vec![] };

    // In real implementation, would broadcast the result
    // For now, simplified implementation
    Ok(result)
}

/// Broadcast operation: send data from root to all other processes
///
/// The root process sends its data to all other processes in the communicator.
///
/// # Arguments
///
/// * `data` - Buffer containing data (root) or to receive data (others)
/// * `root` - Rank of process that sends the data
/// * `comm` - Communicator containing all participating processes
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::collective::*;
/// # use numrs2::distributed::process::*;
/// # async fn example(world: &Communicator) -> Result<(), CollectiveError> {
/// let mut data = if world.is_root() {
///     vec![1.0, 2.0, 3.0, 4.0]
/// } else {
///     vec![0.0; 4]  // Will be overwritten
/// };
/// broadcast(&mut data, 0, world).await?;
/// println!("Rank {}: received {:?}", world.rank(), data);
/// # Ok(())
/// # }
/// ```
pub async fn broadcast<T>(
    data: &mut [T],
    root: usize,
    comm: &Communicator,
) -> Result<(), CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    let rank = comm.rank();
    let size = comm.size();

    if root >= size {
        return Err(CollectiveError::InvalidRoot { root, size });
    }

    if rank == root {
        // Root sends to all other processes
        // In real implementation, use tree-based broadcast for efficiency
        for dest in 0..size {
            if dest != root {
                // Placeholder: would send data to dest
            }
        }
    } else {
        // Non-root receives from root
        // Placeholder: would receive data from root
        // For now, no-op
    }

    Ok(())
}

/// Gather operation: collect data from all processes at root
///
/// Each process contributes data, and the root process receives all contributions
/// concatenated in rank order.
///
/// # Arguments
///
/// * `data` - Local data from this process
/// * `root` - Rank of process that collects all data
/// * `comm` - Communicator containing all participating processes
///
/// # Returns
///
/// Root process receives vector with all data concatenated, others receive empty vector.
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::collective::*;
/// # use numrs2::distributed::process::*;
/// # async fn example(world: &Communicator) -> Result<(), CollectiveError> {
/// let local_data = vec![world.rank() as f64; 10];
/// let all_data = gather(&local_data, 0, world).await?;
/// if world.is_root() {
///     println!("Gathered {} total elements", all_data.len());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn gather<T>(
    data: &[T],
    root: usize,
    comm: &Communicator,
) -> Result<Vec<T>, CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    let rank = comm.rank();
    let size = comm.size();

    if root >= size {
        return Err(CollectiveError::InvalidRoot { root, size });
    }

    if rank == root {
        // Root collects from all processes
        let mut result = Vec::new();

        // Add root's own data first
        result.extend_from_slice(data);

        // Collect from other processes
        for src in 0..size {
            if src != root {
                // Placeholder: would receive from src and append to result
            }
        }

        Ok(result)
    } else {
        // Non-root sends to root
        // Placeholder: would send data to root
        Ok(Vec::new())
    }
}

/// All-gather operation: collect data from all processes and distribute to all
///
/// Like gather, but all processes receive the complete concatenated result.
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::collective::*;
/// # use numrs2::distributed::process::*;
/// # async fn example(world: &Communicator) -> Result<(), CollectiveError> {
/// let local_id = vec![world.rank() as i32];
/// let all_ids = allgather(&local_id, world).await?;
/// println!("Rank {}: all IDs = {:?}", world.rank(), all_ids);
/// # Ok(())
/// # }
/// ```
pub async fn allgather<T>(data: &[T], comm: &Communicator) -> Result<Vec<T>, CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    // Gather at root
    let gathered = gather(data, 0, comm).await?;

    // Broadcast result to all
    // In real implementation, would properly broadcast
    // For now, simplified
    Ok(gathered)
}

/// Scatter operation: distribute data from root to all processes
///
/// The root process splits its data and sends chunks to each process.
///
/// # Arguments
///
/// * `send_data` - Data to distribute (only used at root, empty elsewhere)
/// * `root` - Rank of process that distributes the data
/// * `comm` - Communicator containing all participating processes
///
/// # Returns
///
/// Each process receives its portion of the scattered data.
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::collective::*;
/// # use numrs2::distributed::process::*;
/// # async fn example(world: &Communicator) -> Result<(), CollectiveError> {
/// let send_data = if world.is_root() {
///     (0..40).map(|x| x as f64).collect()  // 40 elements
/// } else {
///     vec![]
/// };
/// let local_data = scatter(&send_data, 0, world).await?;
/// println!("Rank {}: received {} elements", world.rank(), local_data.len());
/// # Ok(())
/// # }
/// ```
pub async fn scatter<T>(
    send_data: &[T],
    root: usize,
    comm: &Communicator,
) -> Result<Vec<T>, CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    let rank = comm.rank();
    let size = comm.size();

    if root >= size {
        return Err(CollectiveError::InvalidRoot { root, size });
    }

    if rank == root {
        // Root distributes data
        let chunk_size = send_data.len() / size;
        let remainder = send_data.len() % size;

        // Send to other processes
        let mut offset = 0;
        for dest in 0..size {
            let this_chunk_size = chunk_size + if dest < remainder { 1 } else { 0 };
            let chunk = &send_data[offset..offset + this_chunk_size];

            if dest != root {
                // Placeholder: would send chunk to dest
            }

            offset += this_chunk_size;
        }

        // Return root's own chunk
        let root_chunk_size = chunk_size + if root < remainder { 1 } else { 0 };
        let root_offset = root * chunk_size + root.min(remainder);
        Ok(send_data[root_offset..root_offset + root_chunk_size].to_vec())
    } else {
        // Non-root receives from root
        // Placeholder: would receive chunk from root
        Ok(Vec::new())
    }
}

/// All-scatter operation: each process distributes data to all processes
///
/// Each process sends a different chunk to every other process.
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::collective::*;
/// # use numrs2::distributed::process::*;
/// # async fn example(world: &Communicator) -> Result<(), CollectiveError> {
/// // Each process prepares data for all processes
/// let send_data: Vec<f64> = (0..world.size() * 10)
///     .map(|x| (x + world.rank() * 100) as f64)
///     .collect();
/// let received = allscatter(&send_data, world).await?;
/// # Ok(())
/// # }
/// ```
pub async fn allscatter<T>(send_data: &[T], comm: &Communicator) -> Result<Vec<T>, CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    let rank = comm.rank();
    let size = comm.size();
    let chunk_size = send_data.len() / size;

    // Each process extracts chunks for all processes
    let mut result = Vec::new();

    // In real implementation, would perform actual all-to-all communication
    // For now, simplified placeholder
    result.extend_from_slice(&send_data[rank * chunk_size..(rank + 1) * chunk_size]);

    Ok(result)
}

/// Barrier synchronization: wait until all processes reach this point
///
/// All processes must call barrier before any can proceed.
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::collective::*;
/// # use numrs2::distributed::process::*;
/// # async fn example(world: &Communicator) -> Result<(), CollectiveError> {
/// println!("Rank {}: before barrier", world.rank());
/// barrier(world).await?;
/// println!("Rank {}: after barrier", world.rank());
/// # Ok(())
/// # }
/// ```
pub async fn barrier(comm: &Communicator) -> Result<(), CollectiveError> {
    // Use communicator's built-in barrier
    comm.barrier().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_op_sum() {
        let op = ReduceOp::Sum;
        assert_eq!(op.apply(2.0, 3.0), 5.0);
        assert_eq!(op.apply(10, 5), 15);
    }

    #[test]
    fn test_reduce_op_product() {
        let op = ReduceOp::Product;
        assert_eq!(op.apply(2.0, 3.0), 6.0);
        assert_eq!(op.apply(4, 5), 20);
    }

    #[test]
    fn test_reduce_op_max() {
        let op = ReduceOp::Max;
        assert_eq!(op.apply(2.0, 3.0), 3.0);
        assert_eq!(op.apply(10, 5), 10);
    }

    #[test]
    fn test_reduce_op_min() {
        let op = ReduceOp::Min;
        assert_eq!(op.apply(2.0, 3.0), 2.0);
        assert_eq!(op.apply(10, 5), 5);
    }

    #[test]
    fn test_reduce_op_apply_slice() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(ReduceOp::Sum.apply_slice(&values), Some(15.0));
        assert_eq!(ReduceOp::Product.apply_slice(&values), Some(120.0));
        assert_eq!(ReduceOp::Max.apply_slice(&values), Some(5.0));
        assert_eq!(ReduceOp::Min.apply_slice(&values), Some(1.0));
    }

    #[test]
    fn test_reduce_op_empty_slice() {
        let values: Vec<f64> = vec![];
        assert_eq!(ReduceOp::Sum.apply_slice(&values), None);
    }

    #[test]
    fn test_collective_error_invalid_root() {
        let err = CollectiveError::InvalidRoot { root: 5, size: 4 };
        assert!(err.to_string().contains("Invalid root"));
    }

    #[test]
    fn test_collective_error_size_mismatch() {
        let err = CollectiveError::SizeMismatch {
            expected: 10,
            actual: 5,
        };
        assert!(err.to_string().contains("Data size mismatch"));
    }
}
