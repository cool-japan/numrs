//! Collective Operations for Distributed Computing
//!
//! This module provides MPI-like collective communication operations for coordinating
//! multiple processes in distributed computations, implemented over the real
//! point-to-point transport in [`super::net`] (via [`Communicator::require_endpoint`]).
//!
//! # Collective Operations
//!
//! - **Reduce**: Combine data from all processes using an operation (sum, max, min, etc.)
//!   — binomial-tree reduction (the exact mirror of [`broadcast`]'s tree).
//! - **All-Reduce**: Reduce and distribute result to all processes — reduce-to-root
//!   followed by broadcast-from-root by default, or a ring reduce-scatter +
//!   allgather for large payloads on enough ranks (see [`allreduce`]'s docs).
//! - **Broadcast**: Send data from one process to all others — binomial tree.
//! - **Gather**: Collect data from all processes at root — linear, rank-ordered.
//! - **All-Gather**: Collect data from all processes and distribute to all — ring.
//! - **Scatter**: Distribute data from root to all processes — NumPy
//!   `array_split`-style block rule.
//! - **All-Scatter**: Distribute data from all to all processes — rotating
//!   pairwise exchange.
//! - **Barrier**: Synchronize all processes — see [`Communicator::barrier`]'s
//!   dissemination algorithm.
//!
//! Every rank participating in a call to any function here must call it (with
//! matching shape where applicable) in the same relative order as every other
//! rank — the same requirement MPI collectives place on their callers. Fixed
//! per-collective wire tags (see the `TAG_*` constants) are safe to reuse
//! across separate calls, and across the ranks that end up in disjoint
//! sub-communicators from one [`Communicator::split`], precisely because of
//! that ordering guarantee plus [`super::net::mailbox::Mailbox`]'s FIFO
//! delivery per `(src, ctx, tag)` key.
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

use super::net::{NetError, SendOpts};
use super::process::{Communicator, ProcessError};
use serde::{Deserialize, Serialize};
use std::ops::{Add, BitAnd, BitOr, Mul};
use thiserror::Error;

/// Errors that can occur during collective operations
#[derive(Error, Debug)]
pub enum CollectiveError {
    #[error("Process error: {0}")]
    Process(#[from] ProcessError),

    /// A failure from the real [`super::net`] transport layer.
    #[error("Transport error: {0}")]
    Net(#[from] NetError),

    #[error("Invalid root rank {root}, must be < {size}")]
    InvalidRoot { root: usize, size: usize },

    #[error("Data size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },

    #[error("Collective operation failed: {0}")]
    OperationFailed(String),

    #[error("Timeout during collective operation")]
    Timeout,
}

impl From<CollectiveError> for NetError {
    /// Best-effort reverse conversion so a [`super::testing::LocalCluster`]
    /// test closure (which must return `Result<T, NetError>`) can run
    /// collectives with plain `?`. A transport failure that already carries
    /// a real [`NetError`] (including one that arrived wrapped in
    /// [`ProcessError::Net`]) unwraps back to it losslessly; anything else
    /// (a logical error like [`CollectiveError::SizeMismatch`]) is
    /// stringified into [`NetError::Io`].
    fn from(err: CollectiveError) -> Self {
        match err {
            CollectiveError::Net(inner) => inner,
            CollectiveError::Process(ProcessError::Net(inner)) => inner,
            other => NetError::Io(other.to_string()),
        }
    }
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
    /// Logical AND (integer types only — see [`Self::reduce_bitwise`])
    And,
    /// Logical OR (integer types only — see [`Self::reduce_bitwise`])
    Or,
}

impl ReduceOp {
    /// Apply the reduction operation to two arithmetic values.
    ///
    /// `And`/`Or` are **not** defined here: `Add`/`Mul`/`PartialOrd` say
    /// nothing about bitwise operators (and this bound admits `f32`/`f64`,
    /// for which bitwise AND/OR have no sensible meaning), so requesting
    /// them is an explicit [`CollectiveError::OperationFailed`] rather than
    /// a silent, meaningless fallback. Use [`Self::reduce_bitwise`] for
    /// integer types instead.
    pub fn apply<T>(&self, a: T, b: T) -> Result<T, CollectiveError>
    where
        T: Add<Output = T> + Mul<Output = T> + PartialOrd + Clone,
    {
        match self {
            ReduceOp::Sum => Ok(a + b),
            ReduceOp::Product => Ok(a * b),
            ReduceOp::Max => Ok(if a > b { a } else { b }),
            ReduceOp::Min => Ok(if a < b { a } else { b }),
            ReduceOp::And | ReduceOp::Or => Err(CollectiveError::OperationFailed(format!(
                "ReduceOp::{self:?} is not defined for arithmetic element types (Add/Mul/PartialOrd); \
                 use ReduceOp::reduce_bitwise for integer types"
            ))),
        }
    }

    /// Apply `And`/`Or` bitwise to two integer values. The mirror image of
    /// [`Self::apply`]: `Sum`/`Product`/`Max`/`Min` are rejected here since
    /// `BitAnd`/`BitOr` alone say nothing about ordering or arithmetic.
    pub fn reduce_bitwise<T>(&self, a: T, b: T) -> Result<T, CollectiveError>
    where
        T: BitAnd<Output = T> + BitOr<Output = T>,
    {
        match self {
            ReduceOp::And => Ok(a & b),
            ReduceOp::Or => Ok(a | b),
            other => Err(CollectiveError::OperationFailed(format!(
                "ReduceOp::{other:?} is not a bitwise operation; use ReduceOp::apply for arithmetic types"
            ))),
        }
    }

    /// Apply reduction to a slice of values.
    pub fn apply_slice<T>(&self, values: &[T]) -> Result<Option<T>, CollectiveError>
    where
        T: Add<Output = T> + Mul<Output = T> + PartialOrd + Clone,
    {
        if values.is_empty() {
            return Ok(None);
        }

        let mut result = values[0].clone();
        for value in &values[1..] {
            result = self.apply(result, value.clone())?;
        }

        Ok(Some(result))
    }
}

// ===========================================================================
// Wire tags
//
// Every collective below gets its own well-separated base, and every
// multi-round algorithm (barrier, the ring variants, all-scatter) folds its
// round/step index into the tag, so no two rounds of one call — and no two
// different collectives sharing one `(comm, ctx)` — can ever be confused for
// each other, independent of the "same relative call order" assumption the
// module docs already require. `TAG_SPLIT_ALLGATHER_BASE` is deliberately
// separate from the public `TAG_ALLGATHER_BASE` so `Communicator::split`'s
// internal `(color, key)` negotiation can never collide with a
// user-initiated `allgather` on the same communicator.
// ===========================================================================
pub(crate) const TAG_BARRIER_BASE: u64 = 0x1_0000_0000;
const TAG_BROADCAST_BASE: u64 = 0x2_0000_0000;
const TAG_REDUCE_BASE: u64 = 0x3_0000_0000;
const TAG_ALLREDUCE_RS_BASE: u64 = 0x4_0000_0000;
const TAG_ALLREDUCE_AG_BASE: u64 = 0x5_0000_0000;
const TAG_GATHER: u64 = 0x6_0000_0000;
const TAG_ALLGATHER_BASE: u64 = 0x7_0000_0000;
const TAG_SCATTER: u64 = 0x8_0000_0000;
const TAG_ALLSCATTER_BASE: u64 = 0x9_0000_0000;
pub(crate) const TAG_SPLIT_ALLGATHER_BASE: u64 = 0xA_0000_0000;

/// `(rank + offset) % size`.
pub(crate) fn mod_add(rank: usize, size: usize, offset: usize) -> usize {
    (rank + offset % size) % size
}

/// `(rank - offset) % size`, computed without unsigned underflow regardless
/// of how large `offset` is.
pub(crate) fn mod_sub(rank: usize, size: usize, offset: usize) -> usize {
    let offset = offset % size;
    (rank + size - offset) % size
}

/// Split `n` items into `size` contiguous block sizes following NumPy's
/// `array_split` rule: the first `n % size` chunks get `ceil(n/size)`
/// elements, the rest get `floor(n/size)`. E.g. `block_split_sizes(10, 4)`
/// is `[3, 3, 2, 2]` (`0..10` → `[0,1,2],[3,4,5],[6,7],[8,9]`).
fn block_split_sizes(n: usize, size: usize) -> Vec<usize> {
    if size == 0 {
        return Vec::new();
    }
    let base = n / size;
    let rem = n % size;
    (0..size)
        .map(|i| if i < rem { base + 1 } else { base })
        .collect()
}

/// `(start, len)` offsets into a flat slice for each of `sizes`, in order.
fn block_split_offsets(sizes: &[usize]) -> Vec<(usize, usize)> {
    let mut offset = 0usize;
    sizes
        .iter()
        .map(|&len| {
            let start = offset;
            offset += len;
            (start, len)
        })
        .collect()
}

fn encode_vec<T: Serialize>(data: &[T]) -> Result<Vec<u8>, CollectiveError> {
    let config = oxicode::config::standard();
    // `encode_to_vec::<T, _>` requires `T: Sized`, which the unsized `[T]`
    // itself never satisfies — pass `&data` (a `&&[T]`, always `Sized`)
    // instead. Serde's blanket `impl<T: ?Sized + Serialize> Serialize for &T`
    // is pure delegation, so this serializes byte-for-byte identically to
    // the slice itself.
    oxicode::serde::encode_to_vec(&data, config).map_err(|e| {
        CollectiveError::OperationFailed(format!("failed to encode collective payload: {e}"))
    })
}

fn decode_vec<T>(bytes: &[u8]) -> Result<Vec<T>, CollectiveError>
where
    T: for<'de> Deserialize<'de>,
{
    let config = oxicode::config::standard();
    let (value, _): (Vec<T>, usize) =
        oxicode::serde::decode_from_slice(bytes, config).map_err(|e| {
            CollectiveError::OperationFailed(format!("failed to decode collective payload: {e}"))
        })?;
    Ok(value)
}

/// Encode `data` and hand it to `comm`'s endpoint for `dst_local` (a rank
/// local to `comm`, translated to the endpoint's global rank space).
///
/// `pub(crate)`: [`super::array::DistributedArray::sync_ghost_cells`] reuses
/// this rather than duplicating the oxicode-over-`Endpoint` plumbing for its
/// own (point-to-point, not collective) boundary exchange.
pub(crate) async fn send_slice<T>(
    comm: &Communicator,
    dst_local: usize,
    tag: u64,
    data: &[T],
) -> Result<(), CollectiveError>
where
    T: Serialize,
{
    let endpoint = comm.require_endpoint()?;
    let dst_global = comm.global_rank(dst_local)?;
    let bytes = encode_vec(data)?;
    endpoint
        .send_owned(
            dst_global,
            comm.context().0,
            tag,
            bytes,
            SendOpts::default(),
        )
        .await?;
    Ok(())
}

/// Receive and decode a `Vec<T>` from `src_local` (a rank local to `comm`).
///
/// `pub(crate)`: see [`send_slice`]'s docs on the one outside caller.
pub(crate) async fn recv_vec<T>(
    comm: &Communicator,
    src_local: usize,
    tag: u64,
) -> Result<Vec<T>, CollectiveError>
where
    T: for<'de> Deserialize<'de>,
{
    let endpoint = comm.require_endpoint()?;
    let src_global = comm.global_rank(src_local)?;
    let bytes = endpoint
        .recv_bytes(src_global, comm.context().0, tag)
        .await?;
    decode_vec(&bytes)
}

fn combine_vecs<T, F>(a: &[T], b: &[T], combine: &F) -> Result<Vec<T>, CollectiveError>
where
    T: Clone,
    F: Fn(T, T) -> Result<T, CollectiveError>,
{
    if a.len() != b.len() {
        return Err(CollectiveError::SizeMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    a.iter()
        .cloned()
        .zip(b.iter().cloned())
        .map(|(x, y)| combine(x, y))
        .collect()
}

/// Binomial-tree broadcast (the classic MPICH algorithm): relabel ranks
/// relative to `root`, then each rank receives at most once (at the mask
/// matching its highest set relative-rank bit) and forwards to descendants
/// at strictly smaller masks. `O(log size)` messages on the critical path.
async fn broadcast_inner<T>(
    data: &mut [T],
    root: usize,
    comm: &Communicator,
    tag_base: u64,
) -> Result<(), CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    let size = comm.size();
    if root >= size {
        return Err(CollectiveError::InvalidRoot { root, size });
    }
    if size <= 1 {
        return Ok(());
    }
    let rank = comm.rank();
    let relative_rank = mod_sub(rank, size, root);

    let mut mask = 1usize;
    while mask < size {
        if relative_rank & mask != 0 {
            // relative_rank & mask != 0 implies relative_rank >= mask.
            let src_rel = relative_rank - mask;
            let src = (src_rel + root) % size;
            let level = mask.trailing_zeros() as u64;
            let incoming: Vec<T> = recv_vec(comm, src, tag_base + level).await?;
            if incoming.len() != data.len() {
                return Err(CollectiveError::SizeMismatch {
                    expected: data.len(),
                    actual: incoming.len(),
                });
            }
            data.clone_from_slice(&incoming);
            break;
        }
        mask <<= 1;
    }

    mask >>= 1;
    while mask > 0 {
        if relative_rank + mask < size {
            // relative_rank + mask < size, and mask strictly decreases every
            // iteration, so this is always a distinct, in-range descendant.
            let dst_rel = relative_rank + mask;
            let dst = (dst_rel + root) % size;
            let level = mask.trailing_zeros() as u64;
            send_slice(comm, dst, tag_base + level, data).await?;
        }
        mask >>= 1;
    }
    Ok(())
}

/// Binomial reduce: the exact mirror of [`broadcast_inner`]'s tree, with
/// roles reversed (receive-and-accumulate at increasing masks, then
/// send-to-parent-and-stop at your own highest set bit). Non-root ranks
/// return an empty vector (existing contract); `combine` lets callers supply
/// either a [`ReduceOp`] or (for [`allreduce_with`]) an arbitrary function.
async fn reduce_inner<T, F>(
    data: &[T],
    combine: &F,
    root: usize,
    comm: &Communicator,
    tag_base: u64,
) -> Result<Vec<T>, CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
    F: Fn(T, T) -> Result<T, CollectiveError>,
{
    let size = comm.size();
    if root >= size {
        return Err(CollectiveError::InvalidRoot { root, size });
    }
    if size <= 1 {
        return Ok(data.to_vec());
    }
    let rank = comm.rank();
    let relative_rank = mod_sub(rank, size, root);
    let mut accum = data.to_vec();

    let mut mask = 1usize;
    while mask < size {
        if relative_rank & mask == 0 {
            let src_rel = relative_rank + mask;
            if src_rel < size {
                let src = (src_rel + root) % size;
                let level = mask.trailing_zeros() as u64;
                let incoming: Vec<T> = recv_vec(comm, src, tag_base + level).await?;
                accum = combine_vecs(&accum, &incoming, combine)?;
            }
        } else {
            // relative_rank & mask != 0 implies relative_rank >= mask, and
            // relative_rank != 0 (root's relative_rank is always 0), so this
            // branch is never taken by root.
            let dst_rel = relative_rank - mask;
            let dst = (dst_rel + root) % size;
            let level = mask.trailing_zeros() as u64;
            send_slice(comm, dst, tag_base + level, &accum).await?;
            return Ok(Vec::new());
        }
        mask <<= 1;
    }
    Ok(accum)
}

/// Ring allgather, generalized to also report each rank's contributed
/// length (used both by the public [`allgather`]/[`allgatherv`] and by
/// [`super::process::Communicator::split`]'s internal `(color, key)`
/// negotiation, under a private `tag_base` — see the module-level tag docs).
pub(crate) async fn allgather_inner<T>(
    data: &[T],
    comm: &Communicator,
    tag_base: u64,
) -> Result<(Vec<T>, Vec<usize>), CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    let size = comm.size();
    if size <= 1 {
        return Ok((data.to_vec(), vec![data.len()]));
    }
    let rank = comm.rank();
    let next = mod_add(rank, size, 1);
    let prev = mod_sub(rank, size, 1);

    let mut have: Vec<Option<Vec<T>>> = vec![None; size];
    have[rank] = Some(data.to_vec());

    for step in 0..size - 1 {
        let send_idx = mod_sub(rank, size, step);
        let recv_idx = mod_sub(rank, size, step + 1);
        let tag = tag_base + step as u64;
        let outgoing = have[send_idx].clone().ok_or_else(|| {
            CollectiveError::OperationFailed("ring allgather: missing chunk to forward".to_string())
        })?;
        send_slice(comm, next, tag, &outgoing).await?;
        let incoming: Vec<T> = recv_vec(comm, prev, tag).await?;
        have[recv_idx] = Some(incoming);
    }

    let mut sizes = Vec::with_capacity(size);
    let mut result = Vec::new();
    for slot in have {
        let chunk = slot.ok_or_else(|| {
            CollectiveError::OperationFailed("ring allgather: incomplete result".to_string())
        })?;
        sizes.push(chunk.len());
        result.extend(chunk);
    }
    Ok((result, sizes))
}

/// Whether [`allreduce`] should use the ring reduce-scatter+allgather path:
/// gated on `size >= 4` and `n >= size` (avoiding the empty-chunk edge case
/// a naive block split would hit for fewer elements than ranks) and an
/// estimated payload of at least 64KiB (below that, tree reduce+broadcast's
/// lower latency wins over the ring's better bandwidth scaling).
fn should_use_ring_allreduce<T>(n: usize, size: usize) -> bool {
    const RING_MIN_BYTES: usize = 64 * 1024;
    size >= 4 && n >= size && n.saturating_mul(std::mem::size_of::<T>()) >= RING_MIN_BYTES
}

/// Ring-based allreduce: reduce-scatter (`size - 1` rounds; after this phase
/// rank `r` holds the complete reduction of chunk `(r + 1) % size`) followed
/// by allgather (`size - 1` more rounds propagating each completed chunk
/// around the ring so every rank ends up with every chunk). `O(size)`
/// messages per rank but each one carries roughly `1/size` of the data,
/// versus the tree path's `O(log size)` messages carrying the *whole*
/// buffer each time — better bandwidth scaling for large payloads.
async fn ring_allreduce<T>(
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
    let size = comm.size();
    let rank = comm.rank();
    let n = data.len();
    let sizes = block_split_sizes(n, size);
    let offsets = block_split_offsets(&sizes);
    let mut chunks: Vec<Vec<T>> = (0..size)
        .map(|i| {
            let (start, len) = offsets[i];
            data[start..start + len].to_vec()
        })
        .collect();

    let next = mod_add(rank, size, 1);
    let prev = mod_sub(rank, size, 1);
    let combine = |a: T, b: T| op.apply(a, b);

    // Reduce-scatter phase.
    for step in 0..size - 1 {
        let send_idx = mod_sub(rank, size, step);
        let recv_idx = mod_sub(rank, size, step + 1);
        let tag = TAG_ALLREDUCE_RS_BASE + step as u64;
        send_slice(comm, next, tag, &chunks[send_idx]).await?;
        let incoming: Vec<T> = recv_vec(comm, prev, tag).await?;
        chunks[recv_idx] = combine_vecs(&chunks[recv_idx], &incoming, &combine)?;
    }

    // Allgather phase: propagate each now-complete chunk the rest of the way
    // around the ring, overwriting (never accumulating).
    for step in 0..size - 1 {
        let send_idx = mod_add(mod_sub(rank, size, step), size, 1);
        let recv_idx = mod_sub(rank, size, step);
        let tag = TAG_ALLREDUCE_AG_BASE + step as u64;
        send_slice(comm, next, tag, &chunks[send_idx]).await?;
        let incoming: Vec<T> = recv_vec(comm, prev, tag).await?;
        chunks[recv_idx] = incoming;
    }

    let mut result = Vec::with_capacity(n);
    for chunk in chunks {
        result.extend(chunk);
    }
    Ok(result)
}

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
/// Root process receives the reduced result; every other process receives an
/// empty vector.
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
    reduce_inner(data, &|a, b| op.apply(a, b), root, comm, TAG_REDUCE_BASE).await
}

/// All-reduce operation: reduce and distribute result to all processes
///
/// Like reduce, but all processes receive the result instead of just the root.
///
/// Uses reduce-to-root-0 followed by broadcast-from-root-0 by default (root's
/// reduced result seeds the broadcast buffer; non-root ranks seed it with
/// `data.to_vec()` purely as a same-length placeholder the broadcast then
/// fully overwrites — this avoids requiring `T: Default`). For large payloads
/// on enough ranks, switches to a ring reduce-scatter+allgather instead (see
/// `should_use_ring_allreduce`), which uses more messages but roughly
/// `1/size` the data per message.
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
    let size = comm.size();
    if size <= 1 {
        return Ok(data.to_vec());
    }
    if should_use_ring_allreduce::<T>(data.len(), size) {
        return ring_allreduce(data, op, comm).await;
    }

    let mut buf = data.to_vec();
    let root_result = reduce_inner(data, &|a, b| op.apply(a, b), 0, comm, TAG_REDUCE_BASE).await?;
    if comm.rank() == 0 {
        buf = root_result;
    }
    broadcast_inner(&mut buf, 0, comm, TAG_BROADCAST_BASE).await?;
    Ok(buf)
}

/// Like [`allreduce`], but combining with an arbitrary function instead of a
/// fixed [`ReduceOp`]. `op` **must be associative and commutative**, exactly
/// as MPI requires of a user-defined reduction operator: the underlying
/// binomial tree feeds already-combined intermediate values back into `op`
/// as ordinary inputs at higher tree levels, in an order that depends on
/// `comm.size()` and each rank's position in the tree — e.g. `|a, b| a.max(b)`
/// is safe (associative+commutative), but `|a, b| (a*a).max(b*b)` is *not*
/// (squaring an already-combined value double-counts it one level up) and
/// will silently produce a result that depends on tree shape rather than
/// only on the multiset of leaf values. Uses the same reduce-then-broadcast
/// shape as the non-ring path of [`allreduce`] (never the ring path, which
/// would only compound the same associativity requirement further).
pub async fn allreduce_with<T, F>(
    data: &[T],
    op: F,
    comm: &Communicator,
) -> Result<Vec<T>, CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
    F: Fn(T, T) -> T,
{
    let size = comm.size();
    if size <= 1 {
        return Ok(data.to_vec());
    }
    let combine = |a: T, b: T| Ok(op(a, b));
    let mut buf = data.to_vec();
    let root_result = reduce_inner(data, &combine, 0, comm, TAG_REDUCE_BASE).await?;
    if comm.rank() == 0 {
        buf = root_result;
    }
    broadcast_inner(&mut buf, 0, comm, TAG_BROADCAST_BASE).await?;
    Ok(buf)
}

/// Reduce-scatter: elementwise-reduce a same-length vector across every
/// rank (via [`allreduce`]), then return only this rank's own
/// `block_split_sizes` slice of the result. Simpler (and, for payloads
/// under the ring threshold, no less efficient) than a true partial
/// reduce-scatter that never materializes the full vector anywhere; see
/// [`allreduce`]'s docs for when the underlying reduction itself already
/// takes the bandwidth-efficient ring path.
pub async fn reduce_scatter<T>(
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
    let size = comm.size();
    let reduced = allreduce(data, op, comm).await?;
    if size <= 1 {
        return Ok(reduced);
    }
    let sizes = block_split_sizes(reduced.len(), size);
    let offsets = block_split_offsets(&sizes);
    let (start, len) = offsets[comm.rank()];
    Ok(reduced[start..start + len].to_vec())
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
    broadcast_inner(data, root, comm, TAG_BROADCAST_BASE).await
}

/// Gather operation: collect data from all processes at root
///
/// Each process contributes data (of any length — lengths need not match
/// across ranks, since each message self-describes its own length), and the
/// root process receives all contributions concatenated in rank order.
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
    let size = comm.size();
    if root >= size {
        return Err(CollectiveError::InvalidRoot { root, size });
    }
    if size <= 1 {
        return Ok(data.to_vec());
    }
    let rank = comm.rank();
    if rank == root {
        let mut result = Vec::new();
        for src in 0..size {
            if src == root {
                result.extend_from_slice(data);
            } else {
                let chunk: Vec<T> = recv_vec(comm, src, TAG_GATHER).await?;
                result.extend(chunk);
            }
        }
        Ok(result)
    } else {
        send_slice(comm, root, TAG_GATHER, data).await?;
        Ok(Vec::new())
    }
}

/// Like [`gather`], but also reports each rank's contributed length at
/// root, so the concatenated boundaries can be recovered (`gather` alone
/// discards them). `None` at every non-root rank.
pub async fn gatherv<T>(
    data: &[T],
    root: usize,
    comm: &Communicator,
) -> Result<Option<(Vec<T>, Vec<usize>)>, CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    let size = comm.size();
    if root >= size {
        return Err(CollectiveError::InvalidRoot { root, size });
    }
    if size <= 1 {
        return Ok(Some((data.to_vec(), vec![data.len()])));
    }
    let rank = comm.rank();
    if rank == root {
        let mut result = Vec::new();
        let mut counts = Vec::with_capacity(size);
        for src in 0..size {
            let chunk: Vec<T> = if src == root {
                data.to_vec()
            } else {
                recv_vec(comm, src, TAG_GATHER).await?
            };
            counts.push(chunk.len());
            result.extend(chunk);
        }
        Ok(Some((result, counts)))
    } else {
        send_slice(comm, root, TAG_GATHER, data).await?;
        Ok(None)
    }
}

/// All-gather operation: collect data from all processes and distribute to all
///
/// Like gather, but all processes receive the complete concatenated result,
/// in rank order. Implemented as a ring: each rank's chunk is forwarded
/// `size - 1` times around the ring until every rank has seen it.
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
    let (result, _sizes) = allgather_inner(data, comm, TAG_ALLGATHER_BASE).await?;
    Ok(result)
}

/// Like [`allgather`], but also returns each rank's contributed length (in
/// rank order) alongside the concatenated result.
pub async fn allgatherv<T>(
    data: &[T],
    comm: &Communicator,
) -> Result<(Vec<T>, Vec<usize>), CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    allgather_inner(data, comm, TAG_ALLGATHER_BASE).await
}

/// Scatter operation: distribute data from root to all processes
///
/// The root process splits its data into `comm.size()` contiguous blocks
/// following NumPy's `array_split` rule (see `block_split_sizes`) and
/// sends one block to each rank — including any that end up empty when
/// there are fewer elements than ranks, which are sent (and received) like
/// any other block rather than skipped.
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
    let size = comm.size();
    if root >= size {
        return Err(CollectiveError::InvalidRoot { root, size });
    }
    if size <= 1 {
        return Ok(send_data.to_vec());
    }
    let rank = comm.rank();
    if rank == root {
        let sizes = block_split_sizes(send_data.len(), size);
        let offsets = block_split_offsets(&sizes);
        let mut own = Vec::new();
        for dst in 0..size {
            let (start, len) = offsets[dst];
            let chunk = &send_data[start..start + len];
            if dst == root {
                own = chunk.to_vec();
            } else {
                send_slice(comm, dst, TAG_SCATTER, chunk).await?;
            }
        }
        Ok(own)
    } else {
        recv_vec(comm, root, TAG_SCATTER).await
    }
}

/// Like [`scatter`], but with explicit, caller-supplied per-rank `counts`
/// instead of the NumPy block rule — the MPI `Scatterv` idiom. `counts` must
/// have exactly `comm.size()` entries and (checked at root) sum to
/// `send_data.len()`.
pub async fn scatterv<T>(
    send_data: &[T],
    counts: &[usize],
    root: usize,
    comm: &Communicator,
) -> Result<Vec<T>, CollectiveError>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + 'static,
{
    let size = comm.size();
    if root >= size {
        return Err(CollectiveError::InvalidRoot { root, size });
    }
    if counts.len() != size {
        return Err(CollectiveError::SizeMismatch {
            expected: size,
            actual: counts.len(),
        });
    }
    let rank = comm.rank();
    if rank == root {
        let total: usize = counts.iter().sum();
        if total != send_data.len() {
            return Err(CollectiveError::SizeMismatch {
                expected: total,
                actual: send_data.len(),
            });
        }
    }
    if size <= 1 {
        return Ok(send_data.to_vec());
    }
    if rank == root {
        let offsets = block_split_offsets(counts);
        let mut own = Vec::new();
        for dst in 0..size {
            let (start, len) = offsets[dst];
            let chunk = &send_data[start..start + len];
            if dst == root {
                own = chunk.to_vec();
            } else {
                send_slice(comm, dst, TAG_SCATTER, chunk).await?;
            }
        }
        Ok(own)
    } else {
        recv_vec(comm, root, TAG_SCATTER).await
    }
}

/// All-scatter operation: each process distributes data to all processes
///
/// `send_data` is split into `comm.size()` blocks (the same NumPy block rule
/// as [`scatter`]); block `j` is destined for rank `j`. A rotating pairwise
/// exchange (`size - 1` rounds; round `s` sends the block destined for
/// `(rank + s) % size` and receives the block destined for this rank from
/// `(rank - s + size) % size`) covers every ordered pair of ranks exactly
/// once. The local result is the concatenation, in *sender* rank order, of
/// what each rank sent this one.
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
    let size = comm.size();
    if size <= 1 {
        return Ok(send_data.to_vec());
    }
    let rank = comm.rank();
    let sizes = block_split_sizes(send_data.len(), size);
    let offsets = block_split_offsets(&sizes);

    let mut received: Vec<Vec<T>> = vec![Vec::new(); size];
    let (self_start, self_len) = offsets[rank];
    received[rank] = send_data[self_start..self_start + self_len].to_vec();

    for step in 1..size {
        let dst = mod_add(rank, size, step);
        let src = mod_sub(rank, size, step);
        let (dstart, dlen) = offsets[dst];
        let outgoing = &send_data[dstart..dstart + dlen];
        let tag = TAG_ALLSCATTER_BASE + step as u64;
        send_slice(comm, dst, tag, outgoing).await?;
        let incoming: Vec<T> = recv_vec(comm, src, tag).await?;
        received[src] = incoming;
    }

    let mut result = Vec::new();
    for chunk in received {
        result.extend(chunk);
    }
    Ok(result)
}

/// Barrier synchronization: wait until all processes reach this point
///
/// All processes must call barrier before any can proceed. Delegates to
/// [`Communicator::barrier`] (dissemination algorithm); see its docs.
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
    comm.barrier().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::process::Communicator;
    use crate::distributed::testing::{ClusterNode, LocalCluster};
    use std::sync::Arc;
    use std::time::Duration;

    fn short_timeout_config() -> super::super::net::EndpointConfig {
        super::super::net::EndpointConfig {
            recv_timeout: Duration::from_secs(2),
            ..super::super::net::EndpointConfig::default()
        }
    }

    async fn run_for_sizes<F, Fut, T>(sizes: &[u32], body: F) -> Vec<(u32, Vec<T>)>
    where
        F: Fn(ClusterNode) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<T, NetError>> + Send + 'static,
        T: Send + 'static,
    {
        let mut out = Vec::with_capacity(sizes.len());
        for &size in sizes {
            let cfg = short_timeout_config();
            let body = body.clone();
            let results =
                LocalCluster::run_connected_with(size, cfg, Duration::from_secs(15), move |node| {
                    body(node)
                })
                .await
                .unwrap_or_else(|e| panic!("collective run failed at world_size={size}: {e}"));
            out.push((size, results));
        }
        out
    }

    // -----------------------------------------------------------------
    // ReduceOp
    // -----------------------------------------------------------------

    #[test]
    fn test_reduce_op_sum() {
        let op = ReduceOp::Sum;
        assert_eq!(op.apply(2.0, 3.0).expect("apply"), 5.0);
        assert_eq!(op.apply(10, 5).expect("apply"), 15);
    }

    #[test]
    fn test_reduce_op_product() {
        let op = ReduceOp::Product;
        assert_eq!(op.apply(2.0, 3.0).expect("apply"), 6.0);
        assert_eq!(op.apply(4, 5).expect("apply"), 20);
    }

    #[test]
    fn test_reduce_op_max() {
        let op = ReduceOp::Max;
        assert_eq!(op.apply(2.0, 3.0).expect("apply"), 3.0);
        assert_eq!(op.apply(10, 5).expect("apply"), 10);
    }

    #[test]
    fn test_reduce_op_min() {
        let op = ReduceOp::Min;
        assert_eq!(op.apply(2.0, 3.0).expect("apply"), 2.0);
        assert_eq!(op.apply(10, 5).expect("apply"), 5);
    }

    #[test]
    fn test_reduce_op_and_or_are_explicit_errors_for_floats() {
        assert!(ReduceOp::And.apply(1.0_f64, 0.0_f64).is_err());
        assert!(ReduceOp::Or.apply(1.0_f64, 0.0_f64).is_err());
    }

    #[test]
    fn test_reduce_bitwise_and_or_on_integers() {
        assert_eq!(
            ReduceOp::And
                .reduce_bitwise(0b1100u32, 0b1010u32)
                .expect("bitwise"),
            0b1000
        );
        assert_eq!(
            ReduceOp::Or
                .reduce_bitwise(0b1100u32, 0b1010u32)
                .expect("bitwise"),
            0b1110
        );
        // Arithmetic ops are rejected the other way round.
        assert!(ReduceOp::Sum.reduce_bitwise(1u32, 2u32).is_err());
    }

    #[test]
    fn test_reduce_op_apply_slice() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(
            ReduceOp::Sum.apply_slice(&values).expect("apply_slice"),
            Some(15.0)
        );
        assert_eq!(
            ReduceOp::Product.apply_slice(&values).expect("apply_slice"),
            Some(120.0)
        );
        assert_eq!(
            ReduceOp::Max.apply_slice(&values).expect("apply_slice"),
            Some(5.0)
        );
        assert_eq!(
            ReduceOp::Min.apply_slice(&values).expect("apply_slice"),
            Some(1.0)
        );
    }

    #[test]
    fn test_reduce_op_empty_slice() {
        let values: Vec<f64> = vec![];
        assert_eq!(
            ReduceOp::Sum.apply_slice(&values).expect("apply_slice"),
            None
        );
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

    // -----------------------------------------------------------------
    // block split helper (NumPy array_split rule)
    // -----------------------------------------------------------------

    #[test]
    fn block_split_matches_numpy_array_split_example() {
        let sizes = block_split_sizes(10, 4);
        assert_eq!(sizes, vec![3, 3, 2, 2]);
        let offsets = block_split_offsets(&sizes);
        let data: Vec<i32> = (0..10).collect();
        let chunks: Vec<Vec<i32>> = offsets
            .iter()
            .map(|&(s, l)| data[s..s + l].to_vec())
            .collect();
        assert_eq!(
            chunks,
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7], vec![8, 9]]
        );
    }

    #[test]
    fn block_split_handles_fewer_elements_than_ranks() {
        let sizes = block_split_sizes(2, 4);
        assert_eq!(sizes, vec![1, 1, 0, 0]);
        let offsets = block_split_offsets(&sizes);
        assert_eq!(offsets, vec![(0, 1), (1, 1), (2, 0), (2, 0)]);
    }

    // -----------------------------------------------------------------
    // barrier / broadcast / reduce / allreduce / gather / allgather /
    // scatter / allscatter, each for every world size the task calls for.
    // -----------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn broadcast_from_nonzero_root_reaches_every_rank() {
        for world_size in 1..=4u32 {
            let root = world_size - 1; // nonzero whenever world_size > 1
            let cfg = short_timeout_config();
            let results = LocalCluster::run_connected_with(
                world_size,
                cfg,
                Duration::from_secs(15),
                move |node: ClusterNode| async move {
                    let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                    let mut buf = if comm.rank() as u32 == root {
                        vec![42.0_f64, 43.0, 44.0]
                    } else {
                        vec![0.0; 3]
                    };
                    broadcast(&mut buf, root as usize, &comm).await?;
                    Ok(buf)
                },
            )
            .await
            .unwrap_or_else(|e| panic!("broadcast failed at world_size={world_size}: {e}"));

            for got in results {
                assert_eq!(got, vec![42.0, 43.0, 44.0], "world_size={world_size}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reduce_non_root_is_empty_and_root_gets_the_sum() {
        for world_size in 1..=4u32 {
            let cfg = short_timeout_config();
            let results = LocalCluster::run_connected_with(
                world_size,
                cfg,
                Duration::from_secs(15),
                |node: ClusterNode| async move {
                    let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                    let local = vec![(comm.rank() + 1) as f64];
                    let result = reduce(&local, ReduceOp::Sum, 0, &comm).await?;
                    Ok((comm.rank(), result))
                },
            )
            .await
            .unwrap_or_else(|e| panic!("reduce failed at world_size={world_size}: {e}"));

            let expected_sum: f64 = (1..=world_size).map(|r| r as f64).sum();
            for (rank, result) in results {
                if rank == 0 {
                    assert_eq!(result, vec![expected_sum], "world_size={world_size}");
                } else {
                    assert_eq!(
                        result,
                        Vec::<f64>::new(),
                        "world_size={world_size}, rank={rank}"
                    );
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn allreduce_sum_p4_all_ones_pinned() {
        let runs = run_for_sizes(&[4], |node: ClusterNode| async move {
            let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
            let local = vec![1.0_f64; 6];
            allreduce(&local, ReduceOp::Sum, &comm)
                .await
                .map_err(NetError::from)
        })
        .await;
        for (_, results) in runs {
            for got in results {
                assert_eq!(got, vec![4.0; 6]);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn allreduce_sum_p4_of_one_to_eight_pinned() {
        let runs = run_for_sizes(&[4], |node: ClusterNode| async move {
            let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
            let local: Vec<f64> = (1..=8).map(|i| i as f64).collect();
            allreduce(&local, ReduceOp::Sum, &comm)
                .await
                .map_err(NetError::from)
        })
        .await;
        let expected: Vec<f64> = (1..=8).map(|i| i as f64 * 4.0).collect();
        for (_, results) in runs {
            for got in results {
                assert_eq!(got, expected);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn allreduce_sum_matches_pinned_vectors_for_every_world_size() {
        let runs = run_for_sizes(&[1, 2, 3, 4], |node: ClusterNode| async move {
            let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
            let local = vec![(comm.rank() + 1) as f64];
            allreduce(&local, ReduceOp::Sum, &comm)
                .await
                .map_err(NetError::from)
        })
        .await;
        for (size, results) in runs {
            let expected: f64 = (1..=size as usize).map(|r| r as f64).sum();
            for got in results {
                assert_eq!(got, vec![expected], "world_size={size}");
            }
        }
    }

    /// The ring reduce-scatter+allgather path is otherwise dead code under
    /// every pinned-vector test above (all use small buffers): this drives
    /// it directly (`p=4`, `n=8192` `f64` clears the 64KiB gate) and checks
    /// it agrees with the plain reduce+broadcast path bit-for-bit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ring_allreduce_matches_tree_allreduce_for_a_large_payload() {
        const N: usize = 8192; // 8192 * 8 bytes = 64KiB, clears the gate at p=4.
        assert!(should_use_ring_allreduce::<f64>(N, 4));

        let cfg = short_timeout_config();
        let ring_results = LocalCluster::run_connected_with(
            4,
            cfg.clone(),
            Duration::from_secs(30),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let local: Vec<f64> = (0..N).map(|i| (i + comm.rank()) as f64).collect();
                allreduce(&local, ReduceOp::Sum, &comm)
                    .await
                    .map_err(NetError::from)
            },
        )
        .await
        .expect("ring allreduce run");

        let tree_results = LocalCluster::run_connected_with(
            4,
            cfg,
            Duration::from_secs(30),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let local: Vec<f64> = (0..N).map(|i| (i + comm.rank()) as f64).collect();
                // Force the tree path with the exact same input shape by
                // calling the internal reduce+broadcast primitives directly.
                let mut buf = local.clone();
                let root_result = reduce_inner(
                    &local,
                    &|a, b| ReduceOp::Sum.apply(a, b),
                    0,
                    &comm,
                    TAG_REDUCE_BASE,
                )
                .await
                .map_err(NetError::from)?;
                if comm.rank() == 0 {
                    buf = root_result;
                }
                broadcast_inner(&mut buf, 0, &comm, TAG_BROADCAST_BASE)
                    .await
                    .map_err(NetError::from)?;
                Ok(buf)
            },
        )
        .await
        .expect("tree allreduce run");

        assert_eq!(ring_results, tree_results);
        let expected: Vec<f64> = (0..N).map(|i| (i * 4 + (0 + 1 + 2 + 3)) as f64).collect();
        assert_eq!(ring_results[0], expected);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gather_is_rank_ordered_and_supports_variable_lengths() {
        for world_size in 1..=4u32 {
            let cfg = short_timeout_config();
            let results = LocalCluster::run_connected_with(
                world_size,
                cfg,
                Duration::from_secs(15),
                |node: ClusterNode| async move {
                    let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                    // Variable length: rank r contributes r+1 elements.
                    let local: Vec<f64> = (0..=comm.rank()).map(|i| i as f64).collect();
                    let gathered = gather(&local, 0, &comm).await?;
                    Ok((comm.rank(), gathered))
                },
            )
            .await
            .unwrap_or_else(|e| panic!("gather failed at world_size={world_size}: {e}"));

            let mut expected = Vec::new();
            for r in 0..world_size as usize {
                expected.extend((0..=r).map(|i| i as f64));
            }
            for (rank, gathered) in results {
                if rank == 0 {
                    assert_eq!(gathered, expected, "world_size={world_size}");
                } else {
                    assert_eq!(gathered, Vec::<f64>::new());
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn allgather_is_rank_ordered_on_every_rank() {
        for world_size in 1..=4u32 {
            let cfg = short_timeout_config();
            let results = LocalCluster::run_connected_with(
                world_size,
                cfg,
                Duration::from_secs(15),
                |node: ClusterNode| async move {
                    let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                    let local = vec![comm.rank() as i64];
                    allgather(&local, &comm).await.map_err(NetError::from)
                },
            )
            .await
            .unwrap_or_else(|e| panic!("allgather failed at world_size={world_size}: {e}"));

            let expected: Vec<i64> = (0..world_size as i64).collect();
            for got in results {
                assert_eq!(got, expected, "world_size={world_size}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scatter_zero_to_ten_over_four_matches_numpy_block_rule() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            4,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let send_data: Vec<i32> = if comm.rank() == 0 {
                    (0..10).collect()
                } else {
                    vec![]
                };
                let local = scatter(&send_data, 0, &comm).await?;
                Ok((comm.rank(), local))
            },
        )
        .await
        .expect("scatter run");

        let expected = [vec![0, 1, 2], vec![3, 4, 5], vec![6, 7], vec![8, 9]];
        for (rank, local) in results {
            assert_eq!(local, expected[rank], "rank {rank}");
        }
    }

    /// Fewer elements than ranks: some ranks must receive (and this must
    /// still *complete*, not hang) an empty chunk.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scatter_fewer_elements_than_ranks_still_delivers_empty_chunks() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            4,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let send_data: Vec<i32> = if comm.rank() == 0 { vec![0, 1] } else { vec![] };
                let local = scatter(&send_data, 0, &comm).await?;
                Ok((comm.rank(), local))
            },
        )
        .await
        .expect("scatter with n<p run");

        let expected = [vec![0], vec![1], vec![], vec![]];
        for (rank, local) in results {
            assert_eq!(local, expected[rank], "rank {rank}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scatter_round_trips_for_every_world_size() {
        for world_size in 1..=4u32 {
            let cfg = short_timeout_config();
            let n = 11usize;
            let results = LocalCluster::run_connected_with(
                world_size,
                cfg,
                Duration::from_secs(15),
                move |node: ClusterNode| async move {
                    let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                    let send_data: Vec<i32> = if comm.rank() == 0 {
                        (0..n as i32).collect()
                    } else {
                        vec![]
                    };
                    let local = scatter(&send_data, 0, &comm).await?;
                    Ok(local)
                },
            )
            .await
            .unwrap_or_else(|e| panic!("scatter failed at world_size={world_size}: {e}"));

            let mut reconstructed: Vec<i32> = Vec::new();
            for chunk in &results {
                reconstructed.extend(chunk);
            }
            let expected: Vec<i32> = (0..n as i32).collect();
            assert_eq!(reconstructed, expected, "world_size={world_size}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn allscatter_delivers_each_ranks_intended_chunk_to_every_peer() {
        for world_size in 1..=4u32 {
            let cfg = short_timeout_config();
            let results = LocalCluster::run_connected_with(
                world_size,
                cfg,
                Duration::from_secs(15),
                |node: ClusterNode| async move {
                    let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                    let size = comm.size();
                    // Rank r's j-th block is a single element encoding (r, j)
                    // as r * 100 + j, so we can verify exactly what arrived
                    // from whom.
                    let send_data: Vec<i64> = (0..size as i64)
                        .map(|j| comm.rank() as i64 * 100 + j)
                        .collect();
                    let got = allscatter(&send_data, &comm).await?;
                    Ok((comm.rank(), got))
                },
            )
            .await
            .unwrap_or_else(|e| panic!("allscatter failed at world_size={world_size}: {e}"));

            let size = world_size as i64;
            for (rank, got) in results {
                let expected: Vec<i64> =
                    (0..size).map(|sender| sender * 100 + rank as i64).collect();
                assert_eq!(got, expected, "world_size={world_size}, rank={rank}");
            }
        }
    }

    // -----------------------------------------------------------------
    // Additive APIs
    // -----------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gatherv_reports_counts_alongside_concatenated_data() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            3,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let local: Vec<i32> = vec![comm.rank() as i32; comm.rank() + 1];
                let out = gatherv(&local, 0, &comm).await?;
                Ok(out)
            },
        )
        .await
        .expect("gatherv run");

        assert_eq!(results[1], None);
        assert_eq!(results[2], None);
        let (data, counts) = results[0].clone().expect("root gets Some");
        assert_eq!(counts, vec![1, 2, 3]);
        assert_eq!(data, vec![0, 1, 1, 2, 2, 2]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scatterv_honors_explicit_counts() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            3,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let counts = vec![1usize, 2, 3];
                let send_data: Vec<i32> = if comm.rank() == 0 {
                    (0..6).collect()
                } else {
                    vec![]
                };
                let local = scatterv(&send_data, &counts, 0, &comm).await?;
                Ok((comm.rank(), local))
            },
        )
        .await
        .expect("scatterv run");

        let expected = [vec![0], vec![1, 2], vec![3, 4, 5]];
        for (rank, local) in results {
            assert_eq!(local, expected[rank]);
        }
    }

    #[tokio::test]
    async fn scatterv_rejects_a_counts_length_mismatch() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            2,
            cfg,
            Duration::from_secs(10),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let counts = vec![1usize]; // wrong length for a world of 2
                let send_data: Vec<i32> = if comm.rank() == 0 { vec![1] } else { vec![] };
                let is_err = scatterv(&send_data, &counts, 0, &comm).await.is_err();
                Ok(is_err)
            },
        )
        .await
        .expect("run");
        assert!(results.iter().all(|&is_err| is_err));
    }

    #[tokio::test]
    async fn scatterv_rejects_a_counts_sum_mismatch_at_root() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            1,
            cfg,
            Duration::from_secs(10),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let counts = vec![5usize]; // does not sum to send_data.len()
                let send_data: Vec<i32> = vec![1, 2, 3];
                let err = scatterv(&send_data, &counts, 0, &comm).await.err();
                let matched = matches!(
                    err,
                    Some(CollectiveError::SizeMismatch {
                        expected: 5,
                        actual: 3
                    })
                );
                Ok(matched)
            },
        )
        .await
        .expect("run");
        assert!(
            results.iter().all(|&matched| matched),
            "expected a SizeMismatch{{expected:5,actual:3}} from scatterv"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn allgatherv_reports_counts_on_every_rank() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            3,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let local: Vec<i32> = vec![comm.rank() as i32; comm.rank() + 1];
                allgatherv(&local, &comm).await.map_err(NetError::from)
            },
        )
        .await
        .expect("allgatherv run");

        for (data, counts) in results {
            assert_eq!(counts, vec![1, 2, 3]);
            assert_eq!(data, vec![0, 1, 1, 2, 2, 2]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn reduce_scatter_gives_each_rank_its_block_of_the_reduction() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            4,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let local: Vec<f64> = (0..10).map(|i| i as f64).collect();
                let my_block = reduce_scatter(&local, ReduceOp::Sum, &comm).await?;
                Ok((comm.rank(), my_block))
            },
        )
        .await
        .expect("reduce_scatter run");

        // Every rank contributed the same [0..10), so the elementwise sum
        // over 4 ranks is [0,4,8,...,36], split via the NumPy block rule.
        let full: Vec<f64> = (0..10).map(|i| i as f64 * 4.0).collect();
        let expected = [
            full[0..3].to_vec(),
            full[3..6].to_vec(),
            full[6..8].to_vec(),
            full[8..10].to_vec(),
        ];
        for (rank, block) in results {
            assert_eq!(block, expected[rank]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn allreduce_with_supports_a_custom_combiner() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            4,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let local = vec![(comm.rank() + 1) as i64];
                // Custom op distinct from any built-in ReduceOp: plain max,
                // chosen because it stays correct regardless of the shape
                // (or intermediate combination order) of the underlying
                // binomial tree -- unlike e.g. "max of squares", which is
                // *not* associative once an already-combined value is fed
                // back in as an input to a later combine step.
                allreduce_with(&local, |a: i64, b: i64| a.max(b), &comm)
                    .await
                    .map_err(NetError::from)
            },
        )
        .await
        .expect("allreduce_with run");
        for got in results {
            assert_eq!(got, vec![4]); // max(1..=4) = 4
        }
    }
}
