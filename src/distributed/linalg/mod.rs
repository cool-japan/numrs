//! Distributed Linear Algebra Operations
//!
//! This module provides distributed implementations of common linear algebra operations
//! for large-scale matrix computations across multiple processes.
//!
//! # Operations
//!
//! Every algorithm below is real and tested against [`super::testing::LocalCluster`]:
//!
//! - [`matrix`]: [`DistributedMatrix`] layouts (row-block, column-block-cyclic),
//!   ring-rotating matrix multiply and allgather-based matrix-vector multiply.
//! - [`householder`]: local (non-distributed) thin Householder QR keeping
//!   *implicit* reflectors — the per-rank factorization step [`mod@tsqr`] uses to
//!   QR-factor each rank's row block.
//! - [`mod@tsqr`]: binary-tree Tall-Skinny QR over row-block matrices
//!   ([`tsqr::tsqr`]), keeping every tree factor so `Q` can be applied in
//!   either direction without ever being formed.
//! - [`decomp`]: decomposition-based orchestration on top of [`mod@tsqr`] and
//!   [`cholesky`] — [`decomp::distributed_qr`], [`decomp::distributed_svd`],
//!   [`decomp::distributed_solve`] and [`decomp::distributed_solve_spd`].
//! - [`cholesky`]: right-looking block-cyclic-column Cholesky
//!   ([`cholesky::block_cholesky`]) plus the distributed forward/back
//!   substitution pair behind [`cholesky::solve_spd`].
//!
//! # Two surfaces, one set of algorithms
//!
//! The names [`distributed_qr`], [`distributed_svd`], [`distributed_solve`],
//! [`distributed_matmul`] and [`distributed_matvec`] appear twice, and the
//! difference is the *operand type*:
//!
//! - the versions in this module take a [`DistributedArray`] — a flat 1-D
//!   partition with no column extent, so it cannot express a distributed
//!   matrix's row/column shape at all. Rather than guess at a layout or
//!   fabricate a numeric result, these permanently return
//!   `Err(`[`DistributedLinalgError::NotImplemented`]`)` naming their
//!   replacement;
//! - the versions in [`decomp`] and [`matrix`] take a [`DistributedMatrix`]
//!   and an explicit [`DistTransport`]. Those are the working
//!   implementations, and are what [`super::prelude`] re-exports under
//!   these same four names plus [`block_cholesky`] — so `use
//!   numrs2::distributed::prelude::*;` reaches the working [`DistributedMatrix`]
//!   versions, not the [`DistributedArray`] stubs above. Reach the stubs
//!   explicitly as `linalg::distributed_qr` and friends if a call site is
//!   still migrating off the old signature.
//!
//! [`distributed_dot`] and [`distributed_norm`] are real over the
//! [`DistributedArray`] surface, routing through the real,
//! point-to-point-backed [`super::collective::allreduce`].
//!
//! # Transport
//!
//! Every algorithm in this module is generic over [`DistTransport`], a
//! point-to-point contract deliberately shaped like
//! [`super::net::Endpoint`]'s frozen `send_bytes`/`recv_bytes` signatures
//! (`(dst|src, ctx, tag, payload)` with per-`(src, ctx, tag)` FIFO
//! delivery). Two implementations ship here:
//!
//! - [`EndpointTransport`], a thin adapter over [`super::net::Endpoint`]:
//!   real TCP links, framing and LZ4, for multi-process runs.
//! - [`LocalTransport`], backed by an in-process [`LocalFabric`] with the
//!   same `(src, ctx, tag)` keying and no sockets, for multi-threaded
//!   single-process runs.
//!
//! Every algorithm here is written once against the trait and exercised
//! on both.
//!
//! The collectives in [`super::collective`] are real, but they are written
//! against [`super::process::Communicator`] specifically. The algorithms in
//! this module are written against the more general [`DistTransport`]
//! instead, so one implementation runs unchanged over a real
//! [`super::net::Endpoint`] (via [`EndpointTransport`]) *or* an in-process
//! [`LocalFabric`] (via [`LocalTransport`]) with no [`Communicator`] or
//! [`super::process::ProcessGroup`] involved at all — useful for
//! multi-threaded single-process numeric work that has no reason to spin up
//! the process/rendezvous machinery. [`bcast_bytes`], [`gather_bytes`],
//! [`allgather_bytes`] and [`allreduce_sum_f64`] below are this module's own
//! `DistTransport`-level collectives, with the same semantics an MPI user
//! expects, that every algorithm here is built from.
//!
//! # Contexts and tags
//!
//! Collective calls take a `ctx` (logical operation id) and a `tag`
//! (phase id within that operation), exactly as
//! [`super::net::mailbox::MailboxKey`] prescribes. Messages under
//! *different* tags have no ordering guarantee relative to each other, so
//! every phase that can be in flight simultaneously uses its own tag. Each
//! public entry point starts by calling [`DistTransport::next_ctx`], which
//! every rank does in lockstep under the SPMD assumption this module makes
//! (all ranks call the same operations in the same order).
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::linalg::*;
//! use numrs2::distributed::array::*;
//! use numrs2::distributed::process::*;
//!
//! # async fn example() -> Result<(), DistributedLinalgError> {
//! let world = init().await.map_err(|e| DistributedLinalgError::LinalgError(e.to_string()))?;
//!
//! // Create distributed vectors
//! let local_a = vec![1.0_f64; 100];
//! let dist_a = DistributedArray::from_local(
//!     local_a,
//!     DistributionStrategy::Block,
//!     400,
//!     &world
//! )?;
//!
//! let local_b = vec![2.0_f64; 100];
//! let dist_b = DistributedArray::from_local(
//!     local_b,
//!     DistributionStrategy::Block,
//!     400,
//!     &world
//! )?;
//!
//! // Distributed dot product
//! let result = distributed_dot(&dist_a, &dist_b).await?;
//! if world.is_root() {
//!     println!("Dot product: {}", result);
//! }
//!
//! finalize(world).await.map_err(|e| DistributedLinalgError::LinalgError(e.to_string()))?;
//! # Ok(())
//! # }
//! ```

pub mod cholesky;
pub mod decomp;
pub mod householder;
pub mod matrix;
pub mod tsqr;

// `decomp`'s entry points are deliberately *not* re-exported here: three of
// them (`distributed_qr`, `distributed_svd`, `distributed_solve`) share a
// name with the legacy `DistributedArray`-surface shims below, and pulling
// both spellings into one namespace would make which one a `use` picked up a
// coin flip. Reach them as `decomp::distributed_qr` and friends; the module
// docs above spell out the split.
pub use cholesky::block_cholesky;
pub use matrix::{DistFloat, DistributedMatrix, Layout};
pub use tsqr::{tsqr, TsqrFactorization, TsqrLevel};

use super::array::{DistributedArray, DistributedArrayError, DistributionStrategy};
use super::collective::{allreduce, CollectiveError, ReduceOp};
use super::net::{Endpoint, NetError, SendOpts};
use super::process::Communicator;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::ops::{Add, Mul};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tokio::sync::Notify;

/// Errors that can occur during distributed linear algebra operations
#[derive(Error, Debug)]
pub enum DistributedLinalgError {
    #[error("Distributed array error: {0}")]
    Array(#[from] DistributedArrayError),

    #[error("Collective operation error: {0}")]
    Collective(#[from] CollectiveError),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Invalid matrix dimensions: rows={rows}, cols={cols}")]
    InvalidDimensions { rows: usize, cols: usize },

    #[error("Singular matrix")]
    SingularMatrix,

    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),

    #[error("Linear algebra error: {0}")]
    LinalgError(String),

    #[error("Not yet implemented: {0}")]
    NotImplemented(String),

    /// A layout the algorithm cannot handle without a different algorithm
    /// entirely — never a "close enough" fallback.
    #[error("Unsupported shape: {0}")]
    UnsupportedShape(String),

    /// The underlying point-to-point transport failed.
    #[error("Transport error: {0}")]
    Transport(String),

    /// A rank id outside `0..world_size`.
    #[error("Invalid rank {rank} (world size {world_size})")]
    InvalidRank { rank: u32, world_size: u32 },

    /// Cholesky met a non-positive pivot: the matrix is not SPD (or lost
    /// definiteness to rounding).
    #[error("Matrix is not positive definite (non-positive pivot at index {index})")]
    NotPositiveDefinite { index: usize },
}

impl From<DistributedLinalgError> for NetError {
    fn from(err: DistributedLinalgError) -> Self {
        match err {
            DistributedLinalgError::UnsupportedShape(msg) => NetError::UnsupportedShape(msg),
            DistributedLinalgError::InvalidRank { rank, world_size } => NetError::InvalidRank {
                rank,
                size: world_size,
            },
            other => NetError::Io(other.to_string()),
        }
    }
}

// ===========================================================================
// Transport contract
// ===========================================================================

/// Point-to-point transport every distributed algorithm in this module is
/// written against.
///
/// The shape mirrors [`super::net::Endpoint`]'s frozen contract:
///
/// - a message is addressed by `(peer, ctx, tag)`;
/// - messages sharing a key are delivered FIFO;
/// - messages under *different* keys have no relative ordering.
///
/// Every `recv_bytes` must be matched by exactly one `send_bytes` with the
/// same `(ctx, tag)` and complementary rank — there is deliberately no
/// wildcard receive, because [`super::net::mailbox::Mailbox`] cannot
/// provide one.
pub trait DistTransport: Send + Sync {
    /// This rank, in `0..world_size()`.
    fn rank(&self) -> u32;

    /// Number of ranks participating.
    fn world_size(&self) -> u32;

    /// Allocate the next logical operation id.
    ///
    /// Under the SPMD assumption (every rank calls the same operations in
    /// the same order) all ranks return the same value here, which is what
    /// keeps two different operations from colliding on one mailbox key.
    fn next_ctx(&self) -> u64;

    /// Send `payload` to `dst` under `(ctx, tag)`.
    fn send_bytes(
        &self,
        dst: u32,
        ctx: u64,
        tag: u64,
        payload: &[u8],
    ) -> impl Future<Output = Result<(), DistributedLinalgError>> + Send;

    /// Receive the next payload queued from `src` under `(ctx, tag)`.
    fn recv_bytes(
        &self,
        src: u32,
        ctx: u64,
        tag: u64,
    ) -> impl Future<Output = Result<Vec<u8>, DistributedLinalgError>> + Send;
}

/// Pending payloads keyed by `(src, ctx, tag)`, FIFO within each key —
/// the same keying [`super::net::mailbox::MailboxKey`] uses.
type FabricQueues = HashMap<(u32, u64, u64), VecDeque<Vec<u8>>>;

/// One rank's inbound mailbox inside a [`LocalFabric`].
#[derive(Debug, Default)]
struct FabricSlot {
    queues: Mutex<FabricQueues>,
    notify: Notify,
}

/// In-process message fabric shared by every rank of a single-process run.
///
/// This is the shared-memory sibling of [`super::net::Endpoint`]: same
/// `(src, ctx, tag)` keying, same FIFO-per-key contract, no sockets. It is
/// what [`super::testing::LocalCluster`]-driven tests run on, and it is
/// also a legitimate production transport for multi-threaded single-process
/// jobs.
#[derive(Debug)]
pub struct LocalFabric {
    slots: Vec<FabricSlot>,
}

impl LocalFabric {
    /// Create a fabric for `world_size` ranks.
    pub fn new(world_size: u32) -> Arc<Self> {
        let mut slots = Vec::with_capacity(world_size as usize);
        for _ in 0..world_size {
            slots.push(FabricSlot::default());
        }
        Arc::new(Self { slots })
    }

    /// Number of ranks this fabric was built for.
    pub fn world_size(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Build the handle rank `rank` uses to talk to everybody else.
    pub fn transport(
        self: &Arc<Self>,
        rank: u32,
    ) -> Result<LocalTransport, DistributedLinalgError> {
        if rank >= self.world_size() {
            return Err(DistributedLinalgError::InvalidRank {
                rank,
                world_size: self.world_size(),
            });
        }
        Ok(LocalTransport {
            fabric: Arc::clone(self),
            rank,
            ctx: AtomicU64::new(1),
        })
    }

    fn push(
        &self,
        dst: u32,
        key: (u32, u64, u64),
        payload: Vec<u8>,
    ) -> Result<(), DistributedLinalgError> {
        let slot = self
            .slots
            .get(dst as usize)
            .ok_or(DistributedLinalgError::InvalidRank {
                rank: dst,
                world_size: self.world_size(),
            })?;
        {
            let mut guard = slot
                .queues
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.entry(key).or_default().push_back(payload);
        }
        // The guard is dropped before waking waiters: this fabric never
        // holds a `std::sync::Mutex` across an `.await`, which is the exact
        // defect `distributed::net` exists to fix in `distributed::comm`.
        slot.notify.notify_waiters();
        Ok(())
    }

    async fn pop(&self, dst: u32, key: (u32, u64, u64)) -> Result<Vec<u8>, DistributedLinalgError> {
        let slot = self
            .slots
            .get(dst as usize)
            .ok_or(DistributedLinalgError::InvalidRank {
                rank: dst,
                world_size: self.world_size(),
            })?;
        loop {
            // Register as a waiter *before* inspecting the queue, so a
            // `notify_waiters` landing between the check and the await
            // cannot be lost.
            let notified = slot.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut guard = slot
                    .queues
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(queue) = guard.get_mut(&key) {
                    if let Some(payload) = queue.pop_front() {
                        return Ok(payload);
                    }
                }
            }
            notified.await;
        }
    }
}

/// One rank's handle onto a [`LocalFabric`].
#[derive(Debug)]
pub struct LocalTransport {
    fabric: Arc<LocalFabric>,
    rank: u32,
    ctx: AtomicU64,
}

impl DistTransport for LocalTransport {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn world_size(&self) -> u32 {
        self.fabric.world_size()
    }

    fn next_ctx(&self) -> u64 {
        self.ctx.fetch_add(1, Ordering::Relaxed)
    }

    async fn send_bytes(
        &self,
        dst: u32,
        ctx: u64,
        tag: u64,
        payload: &[u8],
    ) -> Result<(), DistributedLinalgError> {
        self.fabric
            .push(dst, (self.rank, ctx, tag), payload.to_vec())
    }

    async fn recv_bytes(
        &self,
        src: u32,
        ctx: u64,
        tag: u64,
    ) -> Result<Vec<u8>, DistributedLinalgError> {
        self.fabric.pop(self.rank, (src, ctx, tag)).await
    }
}

/// [`DistTransport`] over the real network stack in [`super::net`].
///
/// This is the multi-process transport: every collective in this module
/// runs unchanged over real TCP links, framing, and LZ4 by handing it an
/// [`Endpoint`] instead of a [`LocalTransport`]. Rank and world size come
/// from the endpoint itself, so the two can never disagree.
#[derive(Debug)]
pub struct EndpointTransport {
    endpoint: Endpoint,
    ctx: AtomicU64,
    opts: SendOpts,
}

impl EndpointTransport {
    /// Wrap a connected [`Endpoint`]. Cloning an endpoint is cheap and
    /// shares one link set, so callers keep using theirs afterwards.
    pub fn new(endpoint: Endpoint) -> Self {
        Self {
            endpoint,
            ctx: AtomicU64::new(1),
            opts: SendOpts::default(),
        }
    }

    /// Permit LZ4 compression for payloads at or above the endpoint's
    /// configured threshold (off by default, matching [`SendOpts`]).
    pub fn with_compression(mut self, compress: bool) -> Self {
        self.opts = SendOpts { compress };
        self
    }

    /// The endpoint this transport sends through.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }
}

impl DistTransport for EndpointTransport {
    fn rank(&self) -> u32 {
        self.endpoint.rank()
    }

    fn world_size(&self) -> u32 {
        self.endpoint.world_size()
    }

    fn next_ctx(&self) -> u64 {
        self.ctx.fetch_add(1, Ordering::Relaxed)
    }

    async fn send_bytes(
        &self,
        dst: u32,
        ctx: u64,
        tag: u64,
        payload: &[u8],
    ) -> Result<(), DistributedLinalgError> {
        self.endpoint
            .send_bytes(dst, ctx, tag, payload, self.opts)
            .await
            .map_err(|e| DistributedLinalgError::Transport(e.to_string()))
    }

    async fn recv_bytes(
        &self,
        src: u32,
        ctx: u64,
        tag: u64,
    ) -> Result<Vec<u8>, DistributedLinalgError> {
        self.endpoint
            .recv_bytes(src, ctx, tag)
            .await
            .map_err(|e| DistributedLinalgError::Transport(e.to_string()))
    }
}

// ===========================================================================
// Byte-level collectives derived from the point-to-point contract
// ===========================================================================

/// Broadcast `payload` (supplied by `root`) to every rank, over a binomial
/// tree: `ceil(log2(p))` rounds instead of `p - 1` root-side sends.
///
/// Every rank receives from exactly one source under `(ctx, tag)`, so one
/// tag suffices for the whole tree.
pub async fn bcast_bytes<C: DistTransport + ?Sized>(
    comm: &C,
    root: u32,
    ctx: u64,
    tag: u64,
    payload: Option<Vec<u8>>,
) -> Result<Vec<u8>, DistributedLinalgError> {
    let size = comm.world_size();
    let rank = comm.rank();
    if root >= size {
        return Err(DistributedLinalgError::InvalidRank {
            rank: root,
            world_size: size,
        });
    }

    let mut data = if rank == root {
        payload.ok_or_else(|| {
            DistributedLinalgError::DimensionMismatch(
                "broadcast root must supply a payload".to_string(),
            )
        })?
    } else {
        Vec::new()
    };
    if size == 1 {
        return Ok(data);
    }

    let vrank = (rank + size - root) % size;
    let mut mask = 1u32;
    while mask < size {
        if vrank & mask != 0 {
            let src = (vrank - mask + root) % size;
            data = comm.recv_bytes(src, ctx, tag).await?;
            break;
        }
        mask <<= 1;
    }
    mask >>= 1;
    while mask > 0 {
        if vrank + mask < size {
            let dst = (vrank + mask + root) % size;
            comm.send_bytes(dst, ctx, tag, &data).await?;
        }
        mask >>= 1;
    }
    Ok(data)
}

/// Broadcast a payload the root computed *fallibly*, so a root-side failure
/// reaches every rank instead of leaving them parked in the broadcast.
///
/// Several algorithms here have a step only one rank can perform — factoring
/// a Cholesky diagonal block, diagonalizing the small `R` of a TSQR, running
/// the triangular solve behind a least-squares fit. Handing those to plain
/// [`bcast_bytes`] turns a *numeric* failure into a *hang*: the root returns
/// `Err` and never sends, while every peer blocks forever waiting for it.
/// This wrapper is what keeps every error path in this module either
/// unanimous by construction (a precondition derived from replicated data)
/// or collectively delivered.
///
/// Wire format: a status byte, then
///
/// - `0`: the payload itself;
/// - `1`: an 8-byte little-endian pivot index, replayed as
///   [`DistributedLinalgError::NotPositiveDefinite`];
/// - `3`: nothing, replayed as [`DistributedLinalgError::SingularMatrix`];
/// - `2`: a UTF-8 message, replayed as
///   [`DistributedLinalgError::LinalgError`].
///
/// The two numeric verdicts a caller is likely to *match on* —
/// "not positive definite" and "singular" — get their own status rather than
/// being flattened into a message, so `matches!(err, SingularMatrix)` works
/// on every rank and not just on the one that did the arithmetic. Any error
/// added here that a caller might branch on needs the same treatment.
///
/// The root round-trips its own frame rather than returning `produce`
/// directly, so a failure is reported identically on every rank. Non-root
/// ranks may pass anything for `produce`; it is never inspected.
pub async fn bcast_fallible_bytes<C: DistTransport + ?Sized>(
    comm: &C,
    root: u32,
    ctx: u64,
    tag: u64,
    produce: Result<Vec<u8>, DistributedLinalgError>,
) -> Result<Vec<u8>, DistributedLinalgError> {
    const STATUS_OK: u8 = 0;
    const STATUS_NOT_POSITIVE_DEFINITE: u8 = 1;
    const STATUS_MESSAGE: u8 = 2;
    const STATUS_SINGULAR: u8 = 3;

    let payload = if comm.rank() == root {
        let mut framed = Vec::new();
        match &produce {
            Ok(bytes) => {
                framed.push(STATUS_OK);
                framed.extend_from_slice(bytes);
            }
            Err(DistributedLinalgError::NotPositiveDefinite { index }) => {
                framed.push(STATUS_NOT_POSITIVE_DEFINITE);
                framed.extend_from_slice(&(*index as u64).to_le_bytes());
            }
            Err(DistributedLinalgError::SingularMatrix) => framed.push(STATUS_SINGULAR),
            Err(other) => {
                framed.push(STATUS_MESSAGE);
                framed.extend_from_slice(other.to_string().as_bytes());
            }
        }
        Some(framed)
    } else {
        None
    };

    let framed = bcast_bytes(comm, root, ctx, tag, payload).await?;
    let (status, body) = framed.split_first().ok_or_else(|| {
        DistributedLinalgError::Transport("fallible broadcast carried no status byte".to_string())
    })?;
    match *status {
        STATUS_OK => Ok(body.to_vec()),
        STATUS_NOT_POSITIVE_DEFINITE => {
            let bytes: [u8; 8] = body.try_into().map_err(|_| {
                DistributedLinalgError::Transport(
                    "fallible broadcast carried a malformed pivot index".to_string(),
                )
            })?;
            Err(DistributedLinalgError::NotPositiveDefinite {
                index: u64::from_le_bytes(bytes) as usize,
            })
        }
        STATUS_SINGULAR => Err(DistributedLinalgError::SingularMatrix),
        STATUS_MESSAGE => Err(DistributedLinalgError::LinalgError(format!(
            "rank {root} failed: {}",
            String::from_utf8_lossy(body)
        ))),
        other => Err(DistributedLinalgError::Transport(format!(
            "fallible broadcast carried an unknown status byte {other}"
        ))),
    }
}

/// Collect every rank's `own` payload at `root`, in rank order.
///
/// Returns `Some(payloads)` on `root` and `None` everywhere else.
pub async fn gather_bytes<C: DistTransport + ?Sized>(
    comm: &C,
    root: u32,
    ctx: u64,
    tag: u64,
    own: &[u8],
) -> Result<Option<Vec<Vec<u8>>>, DistributedLinalgError> {
    let size = comm.world_size();
    let rank = comm.rank();
    if root >= size {
        return Err(DistributedLinalgError::InvalidRank {
            rank: root,
            world_size: size,
        });
    }

    if rank != root {
        comm.send_bytes(root, ctx, tag, own).await?;
        return Ok(None);
    }

    let mut collected = Vec::with_capacity(size as usize);
    for src in 0..size {
        if src == root {
            collected.push(own.to_vec());
        } else {
            collected.push(comm.recv_bytes(src, ctx, tag).await?);
        }
    }
    Ok(Some(collected))
}

/// Every rank ends up with every rank's payload, in rank order, via a ring
/// of `p - 1` steps (each rank forwards what it received last).
///
/// Bandwidth is `O(total_bytes)` per rank rather than the
/// `O(p * total_bytes)` a gather-then-broadcast would push through the
/// root. Each step uses its own tag, since successive steps carry
/// different payloads between the same pair and tags have no cross-tag
/// ordering guarantee.
pub async fn allgather_bytes<C: DistTransport + ?Sized>(
    comm: &C,
    ctx: u64,
    tag: u64,
    own: Vec<u8>,
) -> Result<Vec<Vec<u8>>, DistributedLinalgError> {
    let size = comm.world_size();
    let rank = comm.rank();
    let mut collected = vec![Vec::new(); size as usize];
    let own_index = rank as usize;
    if let Some(slot) = collected.get_mut(own_index) {
        slot.clone_from(&own);
    }
    if size == 1 {
        return Ok(collected);
    }

    let left = (rank + size - 1) % size;
    let right = (rank + 1) % size;
    let mut in_flight = own;
    for step in 0..(size - 1) {
        let step_tag = tag + u64::from(step);
        comm.send_bytes(right, ctx, step_tag, &in_flight).await?;
        let received = comm.recv_bytes(left, ctx, step_tag).await?;
        let owner = ((rank + size - 1 - step) % size) as usize;
        if let Some(slot) = collected.get_mut(owner) {
            slot.clone_from(&received);
        }
        in_flight = received;
    }
    Ok(collected)
}

/// Sum one `f64` across every rank, deterministically: the partial values
/// are allgathered and then summed in rank order, so every rank adds them
/// in the same sequence and gets bit-identical results.
pub async fn allreduce_sum_f64<C: DistTransport + ?Sized>(
    comm: &C,
    ctx: u64,
    tag: u64,
    value: f64,
) -> Result<f64, DistributedLinalgError> {
    let parts = allgather_bytes(comm, ctx, tag, value.to_le_bytes().to_vec()).await?;
    let mut total = 0.0_f64;
    for part in &parts {
        let bytes: [u8; 8] = part.as_slice().try_into().map_err(|_| {
            DistributedLinalgError::Transport(format!(
                "expected 8 bytes for an f64 partial sum, got {}",
                part.len()
            ))
        })?;
        total += f64::from_le_bytes(bytes);
    }
    Ok(total)
}

/// Distributed dot product of two vectors
///
/// Computes the dot product of two distributed vectors using parallel reduction.
///
/// # Arguments
///
/// * `x` - First distributed vector
/// * `y` - Second distributed vector
///
/// # Returns
///
/// The scalar dot product result (same value on all processes)
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::linalg::*;
/// # use numrs2::distributed::array::*;
/// # async fn example(x: &DistributedArray<f64>, y: &DistributedArray<f64>)
/// #     -> Result<(), DistributedLinalgError> {
/// let dot_product = distributed_dot(x, y).await?;
/// println!("Dot product: {}", dot_product);
/// # Ok(())
/// # }
/// ```
pub async fn distributed_dot<T>(
    x: &DistributedArray<T>,
    y: &DistributedArray<T>,
) -> Result<T, DistributedLinalgError>
where
    T: Serialize
        + for<'de> Deserialize<'de>
        + Clone
        + Add<Output = T>
        + Mul<Output = T>
        + PartialOrd
        + Send
        + 'static,
    T: std::iter::Sum,
{
    // Check dimensions
    if x.global_size() != y.global_size() {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "Vector sizes don't match: {} vs {}",
            x.global_size(),
            y.global_size()
        )));
    }

    // Compute local dot product
    let local_x = x.local_data();
    let local_y = y.local_data();

    let local_result = local_x
        .iter()
        .zip(local_y.iter())
        .map(|(a, b)| a.clone() * b.clone())
        .sum::<T>();

    // Global reduction (sum)
    let global_result = allreduce(&[local_result], ReduceOp::Sum, x.comm()).await?;

    global_result
        .into_iter()
        .next()
        .ok_or_else(|| DistributedLinalgError::LinalgError("Empty reduction result".to_string()))
}

/// Distributed matrix-vector multiplication over the legacy
/// [`DistributedArray`] surface.
///
/// The working implementation lives at [`matrix::matvec`], which operates
/// on a [`DistributedMatrix`] and an explicit [`DistTransport`] — the piece
/// of information this legacy signature cannot carry: a [`DistributedArray`]
/// is a flat 1-D partition with no column extent, so it cannot express a
/// matrix's row/column shape at all.
///
/// # Arguments
///
/// * `a` - Distributed matrix (row-distributed)
/// * `x` - Distributed vector
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::linalg::*;
/// # use numrs2::distributed::array::*;
/// # async fn example(a: &DistributedArray<f64>, x: &DistributedArray<f64>)
/// #     -> Result<(), DistributedLinalgError> {
/// let y = distributed_matvec(a, x).await?;
/// # Ok(())
/// # }
/// ```
pub async fn distributed_matvec<T>(
    _a: &DistributedArray<T>,
    _x: &DistributedArray<T>,
) -> Result<DistributedArray<T>, DistributedLinalgError>
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
    Err(DistributedLinalgError::NotImplemented(
        "distributed_matvec over DistributedArray; use matrix::matvec with a DistributedMatrix"
            .to_string(),
    ))
}

/// Distributed matrix multiplication over the legacy [`DistributedArray`]
/// surface.
///
/// The working implementation lives at [`matrix::matmul`] (ring-rotating
/// `B` blocks, `O(k * q / p)` resident memory), which operates on a
/// [`DistributedMatrix`] and an explicit [`DistTransport`]; see
/// [`distributed_matvec`] for why this legacy signature cannot express it.
///
/// # Arguments
///
/// * `a` - First distributed matrix
/// * `b` - Second distributed matrix
///
/// # Example
///
/// ```rust,no_run
/// # use numrs2::distributed::linalg::*;
/// # use numrs2::distributed::array::*;
/// # async fn example(a: &DistributedArray<f64>, b: &DistributedArray<f64>)
/// #     -> Result<(), DistributedLinalgError> {
/// let c = distributed_matmul(a, b).await?;
/// # Ok(())
/// # }
/// ```
pub async fn distributed_matmul<T>(
    _a: &DistributedArray<T>,
    _b: &DistributedArray<T>,
) -> Result<DistributedArray<T>, DistributedLinalgError>
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
    Err(DistributedLinalgError::NotImplemented(
        "distributed_matmul over DistributedArray; use matrix::matmul with a DistributedMatrix"
            .to_string(),
    ))
}

/// Distributed SVD over the legacy [`DistributedArray`] surface.
///
/// `A = U * diag(s) * V^T`. The working implementation lives at
/// [`decomp::distributed_svd`] (TSQR, then a root-side SVD of the small `R`,
/// then `U = Q U_R` through the stored tree); see [`distributed_matvec`] for
/// why this legacy signature cannot express it.
pub async fn distributed_svd<T>(
    _a: &DistributedArray<T>,
) -> Result<(DistributedArray<T>, Vec<T>, DistributedArray<T>), DistributedLinalgError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    Err(DistributedLinalgError::NotImplemented(
        "distributed_svd over DistributedArray; use decomp::distributed_svd with a \
         DistributedMatrix"
            .to_string(),
    ))
}

/// Distributed QR decomposition over the legacy [`DistributedArray`] surface.
///
/// `A = Q * R`. The working implementation lives at
/// [`decomp::distributed_qr`] (a [`tsqr::tsqr`] factorization with `Q`
/// materialized through the stored tree); see [`distributed_matvec`] for why
/// this legacy signature cannot express it.
pub async fn distributed_qr<T>(
    _a: &DistributedArray<T>,
) -> Result<(DistributedArray<T>, DistributedArray<T>), DistributedLinalgError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    Err(DistributedLinalgError::NotImplemented(
        "distributed_qr over DistributedArray; use decomp::distributed_qr with a DistributedMatrix"
            .to_string(),
    ))
}

/// Distributed linear system solve over the legacy [`DistributedArray`]
/// surface.
///
/// Solves `A x = b`. The working implementations live at
/// [`decomp::distributed_solve`] (TSQR least squares, for tall `A`) and
/// [`decomp::distributed_solve_spd`] (Cholesky, for symmetric positive
/// definite `A`); see [`distributed_matvec`] for why this legacy signature
/// cannot express either.
pub async fn distributed_solve<T>(
    _a: &DistributedArray<T>,
    _b: &DistributedArray<T>,
) -> Result<DistributedArray<T>, DistributedLinalgError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    Err(DistributedLinalgError::NotImplemented(
        "distributed_solve over DistributedArray; use decomp::distributed_solve (least squares) \
         or decomp::distributed_solve_spd (Cholesky) with a DistributedMatrix"
            .to_string(),
    ))
}

/// Distributed norm computation
///
/// Computes various norms of a distributed vector.
///
/// Every reduction result is read with `.first()`, never `[0]`, as
/// defensive coding against [`super::collective::allreduce`]'s contract
/// rather than as a workaround for it: today it returns the full,
/// identical result on every rank.
pub async fn distributed_norm<T>(
    x: &DistributedArray<T>,
    p: f64,
) -> Result<f64, DistributedLinalgError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
    T: Into<f64> + Copy,
{
    let local_x = x.local_data();

    let local_sum = if p == f64::INFINITY {
        // Infinity norm: max absolute value
        local_x
            .iter()
            .map(|&v| Into::<f64>::into(v).abs())
            .fold(0.0, f64::max)
    } else if p == 2.0 {
        // L2 norm: sqrt(sum of squares)
        local_x
            .iter()
            .map(|&v| {
                let val = Into::<f64>::into(v);
                val * val
            })
            .sum::<f64>()
    } else {
        // Lp norm: (sum of |x|^p)^(1/p)
        local_x
            .iter()
            .map(|&v| Into::<f64>::into(v).abs().powf(p))
            .sum::<f64>()
    };

    let empty = || {
        DistributedLinalgError::LinalgError(
            "Empty reduction result while computing a distributed norm".to_string(),
        )
    };

    // Global reduction
    let global_sum = if p == f64::INFINITY {
        // Use max reduction for infinity norm
        let result = allreduce(&[local_sum], ReduceOp::Max, x.comm()).await?;
        *result.first().ok_or_else(empty)?
    } else {
        // Use sum reduction for other norms
        let result = allreduce(&[local_sum], ReduceOp::Sum, x.comm()).await?;
        let total = *result.first().ok_or_else(empty)?;
        if p == 2.0 {
            total.sqrt()
        } else {
            total.powf(1.0 / p)
        }
    };

    Ok(global_sum)
}

/// Helper structure for matrix dimensions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MatrixDims {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
}

impl MatrixDims {
    /// Create new matrix dimensions
    pub fn new(rows: usize, cols: usize) -> Result<Self, DistributedLinalgError> {
        if rows == 0 || cols == 0 {
            return Err(DistributedLinalgError::InvalidDimensions { rows, cols });
        }
        Ok(Self { rows, cols })
    }

    /// Check if dimensions are compatible for matrix multiplication
    pub fn can_multiply(&self, other: &MatrixDims) -> bool {
        self.cols == other.rows
    }

    /// Get result dimensions for matrix multiplication
    pub fn multiply_result(&self, other: &MatrixDims) -> Option<MatrixDims> {
        if self.can_multiply(other) {
            Some(MatrixDims {
                rows: self.rows,
                cols: other.cols,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::testing::{LocalCluster, RankContext};

    #[test]
    fn test_matrix_dims() {
        let dims = MatrixDims::new(3, 4).expect("Valid dimensions");
        assert_eq!(dims.rows, 3);
        assert_eq!(dims.cols, 4);
    }

    #[test]
    fn test_matrix_dims_invalid() {
        assert!(MatrixDims::new(0, 4).is_err());
        assert!(MatrixDims::new(3, 0).is_err());
    }

    #[test]
    fn test_matrix_dims_can_multiply() {
        let a = MatrixDims::new(3, 4).expect("Valid");
        let b = MatrixDims::new(4, 5).expect("Valid");
        let c = MatrixDims::new(5, 2).expect("Valid");

        assert!(a.can_multiply(&b));
        assert!(b.can_multiply(&c));
        assert!(!a.can_multiply(&c));
    }

    #[test]
    fn test_matrix_dims_multiply_result() {
        let a = MatrixDims::new(3, 4).expect("Valid");
        let b = MatrixDims::new(4, 5).expect("Valid");

        let result = a.multiply_result(&b).expect("Compatible");
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 5);
    }

    #[test]
    fn test_matrix_dims_multiply_incompatible() {
        let a = MatrixDims::new(3, 4).expect("Valid");
        let b = MatrixDims::new(5, 2).expect("Valid");

        assert!(a.multiply_result(&b).is_none());
    }

    #[test]
    fn fabric_rejects_out_of_range_rank() {
        let fabric = LocalFabric::new(2);
        assert!(matches!(
            fabric.transport(2),
            Err(DistributedLinalgError::InvalidRank {
                rank: 2,
                world_size: 2
            })
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn point_to_point_round_trip_is_fifo_per_key() {
        let fabric = LocalFabric::new(2);
        let results = LocalCluster::run(2, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            async move {
                let comm = fabric.transport(ctx.rank)?;
                if comm.rank() == 0 {
                    comm.send_bytes(1, 7, 1, &[1, 2, 3]).await?;
                    comm.send_bytes(1, 7, 1, &[4, 5]).await?;
                    Ok(Vec::new())
                } else {
                    let first = comm.recv_bytes(0, 7, 1).await?;
                    let second = comm.recv_bytes(0, 7, 1).await?;
                    let mut joined = first;
                    joined.extend_from_slice(&second);
                    Ok(joined)
                }
            }
        })
        .await
        .expect("cluster run should succeed");

        assert_eq!(results[1], vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn binomial_broadcast_reaches_every_rank() {
        for world_size in 1..=8u32 {
            for root in 0..world_size {
                let fabric = LocalFabric::new(world_size);
                let results = LocalCluster::run(world_size, move |ctx: RankContext| {
                    let fabric = Arc::clone(&fabric);
                    async move {
                        let comm = fabric.transport(ctx.rank)?;
                        let ctx_id = comm.next_ctx();
                        let payload = if comm.rank() == root {
                            Some(vec![42u8, 17, 99])
                        } else {
                            None
                        };
                        let got = bcast_bytes(&comm, root, ctx_id, 0, payload).await?;
                        Ok(got)
                    }
                })
                .await
                .expect("cluster run should succeed");

                for got in &results {
                    assert_eq!(
                        got,
                        &vec![42u8, 17, 99],
                        "world_size={world_size} root={root}"
                    );
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ring_allgather_orders_payloads_by_rank() {
        for world_size in 1..=5u32 {
            let fabric = LocalFabric::new(world_size);
            let results = LocalCluster::run(world_size, move |ctx: RankContext| {
                let fabric = Arc::clone(&fabric);
                async move {
                    let comm = fabric.transport(ctx.rank)?;
                    let ctx_id = comm.next_ctx();
                    // Payload length varies with rank, so a size mix-up shows up.
                    let own = vec![comm.rank() as u8; comm.rank() as usize + 1];
                    let all = allgather_bytes(&comm, ctx_id, 0, own).await?;
                    Ok(all)
                }
            })
            .await
            .expect("cluster run should succeed");

            for all in &results {
                assert_eq!(all.len(), world_size as usize);
                for (rank, payload) in all.iter().enumerate() {
                    assert_eq!(payload, &vec![rank as u8; rank + 1]);
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gather_collects_in_rank_order_at_root_only() {
        let fabric = LocalFabric::new(3);
        let results = LocalCluster::run(3, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let ctx_id = comm.next_ctx();
                let own = vec![comm.rank() as u8];
                let gathered = gather_bytes(&comm, 0, ctx_id, 0, &own).await?;
                Ok(gathered)
            }
        })
        .await
        .expect("cluster run should succeed");

        assert_eq!(
            results[0],
            Some(vec![vec![0u8], vec![1u8], vec![2u8]]),
            "root sees every rank in order"
        );
        assert!(results[1].is_none());
        assert!(results[2].is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn allreduce_sum_is_identical_on_every_rank() {
        let fabric = LocalFabric::new(4);
        let results = LocalCluster::run(4, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let ctx_id = comm.next_ctx();
                let total =
                    allreduce_sum_f64(&comm, ctx_id, 0, f64::from(comm.rank()) + 0.5).await?;
                Ok(total)
            }
        })
        .await
        .expect("cluster run should succeed");

        // 0.5 + 1.5 + 2.5 + 3.5
        for total in &results {
            assert_eq!(*total, 8.0);
        }
    }
}
