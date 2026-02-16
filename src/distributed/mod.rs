//! Distributed Computing Module for NumRS2
//!
//! This module provides Pure Rust distributed computing capabilities for large-scale
//! numerical computations across multiple processes and machines.
//!
//! # Overview
//!
//! The distributed module implements MPI-like functionality in Pure Rust (no C bindings),
//! following the COOLJAPAN policy. It provides:
//!
//! - **Communication Layer**: Point-to-point and collective communication using tokio
//! - **Distributed Arrays**: Automatic partitioning and synchronization across processes
//! - **Collective Operations**: Reduce, broadcast, gather, scatter with various strategies
//! - **Distributed Linear Algebra**: Matrix operations distributed across processes
//! - **Process Management**: Process groups and communicators
//! - **Network Optimization**: Topology-aware communication and computation overlap
//! - **Distributed Training**: Data parallelism, model parallelism, and distributed optimizers
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Application Layer                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Distributed Arrays  │  Distributed Linear Algebra              │
//! ├──────────────────────┴──────────────────────────────────────────┤
//! │              Collective Operations                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Process Management  │  Communication Layer                      │
//! ├──────────────────────┴──────────────────────────────────────────┤
//! │              Network Optimization                                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **Pure Rust**: No MPI C bindings, fully safe Rust implementation
//! - **Async Communication**: Built on tokio for efficient async I/O
//! - **Type Safety**: Generic implementations with trait bounds
//! - **Error Handling**: Comprehensive error handling with `Result<T>`
//! - **No Unwrap**: Follows COOLJAPAN no-unwrap policy
//! - **Oxicode Serialization**: Fast binary serialization without C dependencies
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::prelude::*;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize distributed environment
//! let world = init().await?;
//! let rank = world.rank();
//! let size = world.size();
//!
//! // Create distributed array
//! let local_data = vec![rank as f64; 100];
//! let global_size = 400; // Total size across all processes
//! let dist_array = DistributedArray::from_local(
//!     local_data,
//!     DistributionStrategy::Block,
//!     global_size,
//!     &world
//! )?;
//!
//! // Perform collective operation
//! let sum = allreduce(dist_array.local_data(), ReduceOp::Sum, &world).await?;
//!
//! // Distributed matrix multiplication
//! let result = distributed_matmul(&dist_array, &dist_array).await?;
//!
//! // Finalize
//! finalize(world).await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Performance Considerations
//!
//! - Use block distribution for large contiguous arrays
//! - Use cyclic distribution for load balancing irregular workloads
//! - Enable network optimization for topology-aware communication
//! - Overlap computation and communication using async operations
//! - Consider data compression for network-bound operations
//!
//! # See Also
//!
//! - [`comm`]: Low-level communication primitives
//! - [`collective`]: High-level collective operations
//! - [`mod@array`]: Distributed array structures
//! - [`linalg`]: Distributed linear algebra
//! - [`process`]: Process management and communicators

pub mod array;
pub mod collective;
pub mod comm;
pub mod communication;
pub mod coordinator;
pub mod data_parallel;
pub mod linalg;
pub mod model_parallel;
pub mod optimization;
pub mod optimizers;
pub mod process;

/// Re-exports for convenient use
pub mod prelude {
    pub use super::array::{DistributedArray, DistributionStrategy, GlobalIndex, LocalIndex};
    pub use super::collective::{
        allgather, allreduce, allscatter, barrier, broadcast, gather, reduce, scatter, ReduceOp,
    };
    pub use super::comm::{CommunicationChannel, ConnectionManager, Message};
    pub use super::communication::{
        compress_tensor, decompress_tensor, AsyncCommunicator, CompressionStrategy,
        MessagePriority, PipelinedCommunicator, TensorMessage,
    };
    pub use super::coordinator::{
        Checkpoint, DistributedBarrier, ParameterServer, RingAllReduce, TreeAllReduce,
    };
    pub use super::data_parallel::{
        AsyncDataParallel, DistributedDataLoader, GradientAggregation, ShardingStrategy,
        SyncDataParallel,
    };
    pub use super::linalg::{
        distributed_dot, distributed_matmul, distributed_matvec, distributed_qr, distributed_solve,
        distributed_svd,
    };
    pub use super::model_parallel::{
        ActivationCheckpointer, Microbatch, PartitionStrategy, PipelineParallel, PipelineStage,
        TensorParallel,
    };
    pub use super::optimization::{
        optimize_collective, overlap_compute_communicate, NetworkTopology,
    };
    pub use super::optimizers::{DistributedAdam, DistributedSGD};
    pub use super::process::{finalize, init, Communicator, ProcessGroup, WorldCommunicator};
}
