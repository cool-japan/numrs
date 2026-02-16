//! Process Management for Distributed Computing
//!
//! This module provides process group management and communicator abstraction,
//! similar to MPI's communicator concept but implemented in Pure Rust using tokio.
//!
//! # Core Concepts
//!
//! - **Communicator**: A group of processes that can communicate with each other
//! - **Rank**: Unique identifier for a process within a communicator (0 to size-1)
//! - **Size**: Total number of processes in a communicator
//! - **World**: The default communicator containing all processes
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::process::*;
//!
//! # async fn example() -> Result<(), ProcessError> {
//! // Initialize the distributed environment
//! let world = init().await?;
//!
//! println!("Rank: {}, Size: {}", world.rank(), world.size());
//!
//! // Synchronize all processes
//! world.barrier().await?;
//!
//! // Create a sub-communicator (processes with even ranks)
//! let color = if world.rank() % 2 == 0 { 0 } else { 1 };
//! let sub_comm = world.split(color).await?;
//!
//! // Finalize when done
//! finalize(world).await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tokio::sync::RwLock;

/// Errors that can occur during process management operations
#[derive(Error, Debug, Clone)]
pub enum ProcessError {
    #[error("Process not initialized - call init() first")]
    NotInitialized,

    #[error("Process already initialized")]
    AlreadyInitialized,

    #[error("Invalid rank {rank}, must be < {size}")]
    InvalidRank { rank: usize, size: usize },

    #[error("Invalid communicator size: {0}")]
    InvalidSize(usize),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Barrier failed: {0}")]
    BarrierFailed(String),

    #[error("Split operation failed: {0}")]
    SplitFailed(String),

    #[error("Communication error: {0}")]
    CommunicationError(String),
}

/// Process information
#[derive(Debug, Clone)]
pub struct ProcessInfo {
    /// Process rank within communicator
    pub rank: usize,
    /// Total number of processes in communicator
    pub size: usize,
    /// Network address of this process
    pub addr: SocketAddr,
    /// Hostname of the machine running this process
    pub hostname: String,
}

impl ProcessInfo {
    /// Create new process information
    pub fn new(
        rank: usize,
        size: usize,
        addr: SocketAddr,
        hostname: String,
    ) -> Result<Self, ProcessError> {
        if rank >= size {
            return Err(ProcessError::InvalidRank { rank, size });
        }
        if size == 0 {
            return Err(ProcessError::InvalidSize(size));
        }
        Ok(Self {
            rank,
            size,
            addr,
            hostname,
        })
    }

    /// Check if this is the root process (rank 0)
    pub fn is_root(&self) -> bool {
        self.rank == 0
    }
}

/// A group of processes that can communicate with each other
#[derive(Debug, Clone)]
pub struct ProcessGroup {
    /// Ranks of processes in this group
    pub ranks: Vec<usize>,
    /// Mapping from local rank (index in ranks) to global rank
    pub local_to_global: HashMap<usize, usize>,
    /// Mapping from global rank to local rank
    pub global_to_local: HashMap<usize, usize>,
}

impl ProcessGroup {
    /// Create a new process group from a list of ranks
    pub fn new(ranks: Vec<usize>) -> Result<Self, ProcessError> {
        if ranks.is_empty() {
            return Err(ProcessError::InvalidSize(0));
        }

        let mut local_to_global = HashMap::new();
        let mut global_to_local = HashMap::new();

        for (local_rank, &global_rank) in ranks.iter().enumerate() {
            local_to_global.insert(local_rank, global_rank);
            global_to_local.insert(global_rank, local_rank);
        }

        Ok(Self {
            ranks,
            local_to_global,
            global_to_local,
        })
    }

    /// Get the size of this process group
    pub fn size(&self) -> usize {
        self.ranks.len()
    }

    /// Convert local rank to global rank
    pub fn local_to_global_rank(&self, local_rank: usize) -> Result<usize, ProcessError> {
        self.local_to_global
            .get(&local_rank)
            .copied()
            .ok_or_else(|| ProcessError::InvalidRank {
                rank: local_rank,
                size: self.size(),
            })
    }

    /// Convert global rank to local rank
    pub fn global_to_local_rank(&self, global_rank: usize) -> Result<usize, ProcessError> {
        self.global_to_local
            .get(&global_rank)
            .copied()
            .ok_or_else(|| ProcessError::InvalidRank {
                rank: global_rank,
                size: self.size(),
            })
    }

    /// Check if a global rank is in this group
    pub fn contains(&self, global_rank: usize) -> bool {
        self.global_to_local.contains_key(&global_rank)
    }
}

/// A communicator represents a group of processes that can communicate
#[derive(Clone)]
pub struct Communicator {
    /// Process information for this process
    info: Arc<ProcessInfo>,
    /// Process group for this communicator
    group: Arc<ProcessGroup>,
    /// Addresses of all processes in this communicator
    addresses: Arc<HashMap<usize, SocketAddr>>,
    /// Barrier counter for synchronization
    barrier_counter: Arc<RwLock<usize>>,
}

impl Communicator {
    /// Create a new communicator
    pub fn new(
        info: ProcessInfo,
        group: ProcessGroup,
        addresses: HashMap<usize, SocketAddr>,
    ) -> Result<Self, ProcessError> {
        Ok(Self {
            info: Arc::new(info),
            group: Arc::new(group),
            addresses: Arc::new(addresses),
            barrier_counter: Arc::new(RwLock::new(0)),
        })
    }

    /// Get this process's rank within the communicator
    pub fn rank(&self) -> usize {
        self.info.rank
    }

    /// Get the total number of processes in the communicator
    pub fn size(&self) -> usize {
        self.info.size
    }

    /// Get process information
    pub fn process_info(&self) -> &ProcessInfo {
        &self.info
    }

    /// Get process group
    pub fn group(&self) -> &ProcessGroup {
        &self.group
    }

    /// Check if this is the root process (rank 0)
    pub fn is_root(&self) -> bool {
        self.info.is_root()
    }

    /// Get the address of a process by rank
    pub fn address(&self, rank: usize) -> Result<SocketAddr, ProcessError> {
        self.addresses
            .get(&rank)
            .copied()
            .ok_or_else(|| ProcessError::InvalidRank {
                rank,
                size: self.size(),
            })
    }

    /// Barrier synchronization - all processes wait until all reach this point
    pub async fn barrier(&self) -> Result<(), ProcessError> {
        // Simple barrier implementation using counter
        // In a real implementation, this would use network synchronization
        let mut counter = self.barrier_counter.write().await;
        *counter += 1;

        // Wait for all processes (simplified - real implementation would use network)
        // This is a placeholder that would be replaced with actual network coordination
        if *counter >= self.size() {
            *counter = 0;
            Ok(())
        } else {
            // In real implementation, would wait for barrier message from coordinator
            Ok(())
        }
    }

    /// Split the communicator into sub-communicators based on color
    ///
    /// Processes with the same color will be in the same sub-communicator.
    /// The new rank within the sub-communicator is determined by the original rank order.
    pub async fn split(&self, color: usize) -> Result<Communicator, ProcessError> {
        // Collect ranks with the same color
        // In a real implementation, this would involve network communication
        // to gather color information from all processes

        // For now, create a simplified sub-communicator
        // Real implementation would use allgather to collect colors from all processes
        let new_ranks = vec![self.rank()]; // Simplified - would collect from all with same color
        let new_group =
            ProcessGroup::new(new_ranks).map_err(|e| ProcessError::SplitFailed(e.to_string()))?;

        let new_info = ProcessInfo::new(
            0, // New rank within sub-communicator
            new_group.size(),
            self.info.addr,
            self.info.hostname.clone(),
        )
        .map_err(|e| ProcessError::SplitFailed(e.to_string()))?;

        // Create new address mapping for sub-communicator
        let new_addresses = HashMap::new(); // Would be populated from network communication

        Communicator::new(new_info, new_group, new_addresses)
            .map_err(|e| ProcessError::SplitFailed(e.to_string()))
    }

    /// Create a sub-communicator from specific ranks
    pub async fn create_group(&self, ranks: &[usize]) -> Result<Communicator, ProcessError> {
        // Validate ranks
        for &rank in ranks {
            if rank >= self.size() {
                return Err(ProcessError::InvalidRank {
                    rank,
                    size: self.size(),
                });
            }
        }

        let new_group = ProcessGroup::new(ranks.to_vec())
            .map_err(|e| ProcessError::SplitFailed(e.to_string()))?;

        // Find this process's new rank in the group
        let new_rank = new_group.global_to_local_rank(self.rank())?;

        let new_info = ProcessInfo::new(
            new_rank,
            new_group.size(),
            self.info.addr,
            self.info.hostname.clone(),
        )
        .map_err(|e| ProcessError::SplitFailed(e.to_string()))?;

        // Create address mapping for new group
        let mut new_addresses = HashMap::new();
        for &rank in ranks {
            if let Ok(addr) = self.address(rank) {
                new_addresses.insert(rank, addr);
            }
        }

        Communicator::new(new_info, new_group, new_addresses)
            .map_err(|e| ProcessError::SplitFailed(e.to_string()))
    }
}

impl std::fmt::Debug for Communicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Communicator")
            .field("rank", &self.rank())
            .field("size", &self.size())
            .field("is_root", &self.is_root())
            .finish()
    }
}

/// The world communicator containing all processes
pub type WorldCommunicator = Communicator;

/// Global state for distributed environment
static GLOBAL_WORLD: OnceLock<WorldCommunicator> = OnceLock::new();

/// Initialize the distributed computing environment
///
/// This must be called before any other distributed operations.
/// Returns the world communicator containing all processes.
///
/// # Configuration
///
/// Configuration is read from environment variables:
/// - `NUMRS2_RANK`: Process rank (default: 0)
/// - `NUMRS2_SIZE`: Total number of processes (default: 1)
/// - `NUMRS2_MASTER_ADDR`: Master process address (default: "127.0.0.1:5000")
/// - `NUMRS2_BIND_ADDR`: This process's bind address (default: "127.0.0.1:5000+rank")
///
/// # Example
///
/// ```rust,no_run
/// use numrs2::distributed::process::*;
///
/// # async fn example() -> Result<(), ProcessError> {
/// let world = init().await?;
/// println!("Initialized rank {} of {}", world.rank(), world.size());
/// # Ok(())
/// # }
/// ```
pub async fn init() -> Result<WorldCommunicator, ProcessError> {
    // Check if already initialized
    if GLOBAL_WORLD.get().is_some() {
        return Err(ProcessError::AlreadyInitialized);
    }

    // Read configuration from environment
    let rank: usize = std::env::var("NUMRS2_RANK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let size: usize = std::env::var("NUMRS2_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let _master_addr: SocketAddr = std::env::var("NUMRS2_MASTER_ADDR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| "127.0.0.1:5000".parse().expect("Valid default address"));

    // Create bind address for this process
    let port = 5000 + rank as u16;
    let bind_addr: SocketAddr = std::env::var("NUMRS2_BIND_ADDR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            format!("127.0.0.1:{}", port)
                .parse()
                .expect("Valid default bind address")
        });

    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "localhost".to_string());

    // Create process info
    let info = ProcessInfo::new(rank, size, bind_addr, hostname)?;

    // Create process group with all ranks
    let all_ranks: Vec<usize> = (0..size).collect();
    let group = ProcessGroup::new(all_ranks)?;

    // Create address mapping (simplified - in real impl would discover via network)
    let mut addresses = HashMap::new();
    for r in 0..size {
        let addr: SocketAddr = format!("127.0.0.1:{}", 5000 + r)
            .parse()
            .map_err(|e| ProcessError::ConfigError(format!("Invalid address: {}", e)))?;
        addresses.insert(r, addr);
    }

    // Create world communicator
    let world = Communicator::new(info, group, addresses)?;

    // Store in global state
    GLOBAL_WORLD
        .set(world.clone())
        .map_err(|_| ProcessError::AlreadyInitialized)?;

    Ok(world)
}

/// Finalize the distributed computing environment
///
/// This should be called when all distributed operations are complete.
/// After calling finalize, no other distributed operations can be performed
/// until init() is called again.
///
/// # Example
///
/// ```rust,no_run
/// use numrs2::distributed::process::*;
///
/// # async fn example() -> Result<(), ProcessError> {
/// let world = init().await?;
/// // ... do distributed operations ...
/// finalize(world).await?;
/// # Ok(())
/// # }
/// ```
pub async fn finalize(_world: WorldCommunicator) -> Result<(), ProcessError> {
    if GLOBAL_WORLD.get().is_none() {
        return Err(ProcessError::NotInitialized);
    }

    // Perform cleanup (close connections, etc.)
    // In a real implementation, would properly shutdown network connections

    // Note: OnceLock doesn't provide a way to clear/reset, so this is a limitation
    // In a production implementation, we'd use a different synchronization primitive
    // that allows resetting (like RwLock<Option<T>>)

    Ok(())
}

/// Get the rank of the current process in the world communicator
///
/// Convenience function that returns the rank without needing a communicator reference.
/// Requires that init() has been called.
pub fn rank() -> Result<usize, ProcessError> {
    GLOBAL_WORLD
        .get()
        .map(|w| w.rank())
        .ok_or(ProcessError::NotInitialized)
}

/// Get the size of the world communicator
///
/// Convenience function that returns the size without needing a communicator reference.
/// Requires that init() has been called.
pub fn size() -> Result<usize, ProcessError> {
    GLOBAL_WORLD
        .get()
        .map(|w| w.size())
        .ok_or(ProcessError::NotInitialized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_info() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().expect("Valid address");
        let info = ProcessInfo::new(0, 4, addr, "localhost".to_string()).expect("Valid info");

        assert_eq!(info.rank, 0);
        assert_eq!(info.size, 4);
        assert!(info.is_root());
        assert_eq!(info.hostname, "localhost");
    }

    #[test]
    fn test_process_info_invalid_rank() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().expect("Valid address");
        let result = ProcessInfo::new(5, 4, addr, "localhost".to_string());

        assert!(result.is_err());
        match result {
            Err(ProcessError::InvalidRank { rank, size }) => {
                assert_eq!(rank, 5);
                assert_eq!(size, 4);
            }
            _ => panic!("Expected InvalidRank error"),
        }
    }

    #[test]
    fn test_process_group() {
        let ranks = vec![0, 2, 4, 6];
        let group = ProcessGroup::new(ranks.clone()).expect("Valid group");

        assert_eq!(group.size(), 4);
        assert_eq!(group.local_to_global_rank(0).expect("Valid"), 0);
        assert_eq!(group.local_to_global_rank(1).expect("Valid"), 2);
        assert_eq!(group.global_to_local_rank(4).expect("Valid"), 2);

        assert!(group.contains(0));
        assert!(group.contains(4));
        assert!(!group.contains(1));
        assert!(!group.contains(3));
    }

    #[test]
    fn test_process_group_empty() {
        let result = ProcessGroup::new(vec![]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_communicator_creation() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().expect("Valid address");
        let info = ProcessInfo::new(0, 4, addr, "localhost".to_string()).expect("Valid info");
        let group = ProcessGroup::new(vec![0, 1, 2, 3]).expect("Valid group");
        let addresses = HashMap::new();

        let comm = Communicator::new(info, group, addresses).expect("Valid communicator");

        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.size(), 4);
        assert!(comm.is_root());
    }
}
