//! Process Management for Distributed Computing
//!
//! This module provides process group management and communicator abstraction,
//! similar to MPI's communicator concept, built on the real point-to-point
//! transport in [`super::net`].
//!
//! # Core Concepts
//!
//! - **Communicator**: A group of processes that can communicate with each other.
//!   Every communicator holds an `Option<Arc<`[`Endpoint`]`>>` (`None` only for
//!   the offline/legacy [`Communicator::new`] constructor — see its docs) plus a
//!   [`ContextId`]. Sub-communicators produced by [`Communicator::split`] and
//!   [`Communicator::create_group`] share the *same* `Arc<Endpoint>` (no new
//!   sockets, no new mesh) and are distinguished purely by `ContextId`.
//! - **Rank**: Unique identifier for a process within a communicator (0 to size-1).
//! - **Size**: Total number of processes in a communicator.
//! - **World**: The default communicator containing all processes, held in
//!   `GLOBAL_WORLD` between [`init`] and [`finalize`].
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use thiserror::Error;

use super::bootstrap::{self, RendezvousMode};
use super::collective;
use super::net::{Endpoint, EndpointConfig, NetError, SendOpts};

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

    /// A failure from the real [`super::net`] transport layer (connect,
    /// send, recv, bootstrap, ...). `NetError` is `Clone`, so this preserves
    /// `ProcessError`'s own `Clone` derive.
    #[error("Transport error: {0}")]
    Net(#[from] NetError),
}

impl From<ProcessError> for NetError {
    /// Best-effort reverse conversion, used by test harnesses (e.g.
    /// [`super::testing::LocalCluster`] closures, which must return
    /// `Result<T, NetError>`) that build a [`Communicator`] and then run
    /// collectives against it with `?`. A [`ProcessError::Net`] unwraps back
    /// to its original [`NetError`] losslessly; everything else is
    /// stringified into [`NetError::Io`].
    fn from(err: ProcessError) -> Self {
        match err {
            ProcessError::Net(inner) => inner,
            other => NetError::Io(other.to_string()),
        }
    }
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

/// A group of processes that can communicate with each other.
///
/// `ranks` are always in the *endpoint's* global rank space: for the world
/// communicator that is simply `0..world_size`, and for a sub-communicator
/// produced by [`Communicator::split`]/[`Communicator::create_group`] it is
/// whichever subset of the parent's global ranks ended up as members. This
/// is exactly what lets a sub-communicator share its parent's `Arc<Endpoint>`
/// (whose rank numbering never changes) while still letting collectives
/// address peers by the sub-communicator's own local rank.
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

/// Opaque logical context id distinguishing sub-communicators that share one
/// physical [`Endpoint`]. Threaded through every [`Endpoint::send_bytes`]/
/// [`Endpoint::recv_bytes`] call as the wire `ctx` field, so two
/// sub-communicators derived from the same world never confuse each other's
/// traffic even though messages travel over the same TCP links.
///
/// Never equal to `u64::MAX`: that value is
/// [`super::net::frame::CTX_CONTROL`], reserved for transport-internal
/// control frames and rejected by [`Endpoint::send_bytes`] for user traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextId(pub u64);

impl ContextId {
    /// The context every top-level "world" communicator uses.
    pub const WORLD: ContextId = ContextId(0);

    /// Derive a child context from `parent` and a locally-incrementing
    /// sequence number, via a single splitmix64 mixing step (Vigna's
    /// finalizer) over `parent.0` combined with `seq`.
    ///
    /// `seq` must come from a counter that lives on the *parent*
    /// communicator (see [`Communicator::split`]'s docs on `split_seq`), not
    /// on the endpoint: every rank participating in one logical split
    /// independently calls this with the *same* `parent` and the *same*
    /// `seq` (because SPMD code calls `split`/`create_group` the same number
    /// of prior times, in the same order, on every rank), so every member of
    /// the resulting sub-communicator agrees on the new `ContextId` without
    /// any network round trip to negotiate it. Two colors produced by one
    /// `split` call safely reuse the *same* derived context: colors
    /// partition the parent's ranks, so no physical rank ever belongs to two
    /// groups sharing a context at once.
    fn derive(parent: ContextId, seq: u64) -> ContextId {
        const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
        let seed = parent
            .0
            .wrapping_add(seq.wrapping_add(1).wrapping_mul(GAMMA));
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // ctx == u64::MAX is CTX_CONTROL, reserved for transport control
        // frames; never hand it out as a user-visible context.
        if z == u64::MAX {
            z ^= 1;
        }
        ContextId(z)
    }
}

/// A communicator represents a group of processes that can communicate.
///
/// See the module docs for how `endpoint`/`context` relate: sub-communicators
/// share one `Arc<Endpoint>` and differ only by `ContextId` (plus the
/// `group`'s local↔global rank mapping into that shared endpoint's rank
/// space).
#[derive(Clone)]
pub struct Communicator {
    /// Process information for this process, *local* to this communicator
    /// (i.e. `info.rank`/`info.size` are this communicator's own rank/size,
    /// not necessarily the underlying endpoint's).
    info: Arc<ProcessInfo>,
    /// Process group for this communicator: local rank ↔ endpoint-global rank.
    group: Arc<ProcessGroup>,
    /// Best-effort address book (legacy metadata for [`Self::address`]).
    /// Real sends never consult this — they go through `endpoint` by global
    /// rank — so it is safe for this to be incomplete or empty.
    addresses: Arc<HashMap<usize, SocketAddr>>,
    /// The shared real transport. `None` only for a communicator built via
    /// the offline [`Communicator::new`] constructor (kept for the several
    /// call sites elsewhere in this crate that build a structural,
    /// size-one, never-actually-communicating `Communicator` synchronously
    /// in a plain `#[test]` with no tokio runtime available to bind a real
    /// socket). Every collective in [`super::collective`] and every method
    /// here short-circuits at `size() == 1` *before* touching the endpoint,
    /// so an offline communicator works for exactly the cases those call
    /// sites need and errors clearly ([`Self::require_endpoint`]) for
    /// anything that would actually need the network.
    endpoint: Option<Arc<Endpoint>>,
    /// This communicator's logical context on the shared endpoint.
    context: ContextId,
    /// Parent-side counter for deriving child contexts in
    /// [`Self::split`]/[`Self::create_group`]. Lives on *this* communicator
    /// (shared only by its own clones via the `Arc`), never on the endpoint
    /// — an endpoint-global counter would let two unrelated, simultaneous
    /// splits (of two different communicators sharing one endpoint) race
    /// against each other and disagree on the sequence number.
    split_seq: Arc<AtomicU64>,
}

impl Communicator {
    /// Build an offline communicator directly from process/group/address
    /// data, with **no attached transport** (`endpoint` is `None`).
    ///
    /// This is a synchronous constructor — building a real [`Endpoint`]
    /// requires binding a socket, which is inherently async — so it cannot
    /// itself stand up real networking. It exists for callers that need a
    /// structurally valid, rank/size-correct `Communicator` without a tokio
    /// runtime (several plain `#[test]` helpers elsewhere in this crate
    /// build a size-one `Communicator` this way purely to exercise
    /// non-networked code paths). Every collective operation still works
    /// correctly on such a communicator as long as `size() <= 1` (nothing to
    /// talk to); anything that would need to actually send/receive returns
    /// [`ProcessError::CommunicationError`] via [`Self::require_endpoint`]
    /// rather than panicking.
    ///
    /// Real distributed use should go through [`init`] or
    /// [`Self::from_endpoint`] instead.
    pub fn new(
        info: ProcessInfo,
        group: ProcessGroup,
        addresses: HashMap<usize, SocketAddr>,
    ) -> Result<Self, ProcessError> {
        Ok(Self {
            info: Arc::new(info),
            group: Arc::new(group),
            addresses: Arc::new(addresses),
            endpoint: None,
            context: ContextId::WORLD,
            split_seq: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Build the world communicator directly from an already-connected
    /// [`Endpoint`] (rank/size/local address are read straight off it, and
    /// the group is the full `0..world_size`). This is the real
    /// constructor: [`init`] uses it after [`bootstrap::bootstrap`], and
    /// [`super::testing::LocalCluster::run_connected`] harnesses use it to
    /// turn a [`super::testing::ClusterNode`]'s endpoint into a
    /// `Communicator` collectives can run against.
    pub fn from_endpoint(endpoint: Arc<Endpoint>) -> Result<Self, ProcessError> {
        let rank = endpoint.rank() as usize;
        let size = endpoint.world_size() as usize;
        let group = ProcessGroup::new((0..size).collect())?;
        let info = ProcessInfo::new(rank, size, endpoint.local_addr(), resolve_hostname())?;
        Ok(Self {
            info: Arc::new(info),
            group: Arc::new(group),
            addresses: Arc::new(HashMap::new()),
            endpoint: Some(endpoint),
            context: ContextId::WORLD,
            split_seq: Arc::new(AtomicU64::new(0)),
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

    /// This communicator's logical context on its shared endpoint.
    pub fn context(&self) -> ContextId {
        self.context
    }

    /// The shared transport, if any (`None` only for a
    /// [`Self::new`]-built offline communicator).
    pub fn endpoint(&self) -> Option<&Arc<Endpoint>> {
        self.endpoint.as_ref()
    }

    /// [`Self::endpoint`], or a descriptive error when this communicator has
    /// none attached. Every real send/recv path goes through this.
    pub fn require_endpoint(&self) -> Result<&Arc<Endpoint>, ProcessError> {
        self.endpoint.as_ref().ok_or_else(|| {
            ProcessError::CommunicationError(
                "this communicator has no attached transport (it was built via the offline \
                 Communicator::new constructor rather than init()/from_endpoint); only \
                 size() <= 1 operations are supported without a real endpoint"
                    .to_string(),
            )
        })
    }

    /// Translate `local_rank` (within this communicator) to the underlying
    /// endpoint's global rank, via [`Self::group`].
    pub fn global_rank(&self, local_rank: usize) -> Result<u32, ProcessError> {
        let global = self.group.local_to_global_rank(local_rank)?;
        u32::try_from(global).map_err(|_| ProcessError::InvalidRank {
            rank: local_rank,
            size: self.size(),
        })
    }

    /// Get the address of a process by rank (best-effort metadata; see
    /// `addresses`'s docs — real sends never consult this).
    pub fn address(&self, rank: usize) -> Result<SocketAddr, ProcessError> {
        self.addresses
            .get(&rank)
            .copied()
            .ok_or_else(|| ProcessError::InvalidRank {
                rank,
                size: self.size(),
            })
    }

    /// Derive this communicator's next child [`ContextId`] and advance
    /// [`Self::split_seq`]. Shared logic for [`Self::split`] and
    /// [`Self::create_group`].
    fn derive_child_context(&self) -> ContextId {
        let seq = self.split_seq.fetch_add(1, Ordering::SeqCst);
        ContextId::derive(self.context, seq)
    }

    /// Barrier synchronization: every rank blocks here until every other
    /// rank in the communicator has also called `barrier`.
    ///
    /// Implements the dissemination algorithm (Hensgen, Finkel & Manber
    /// 1988): `ceil(log2(size))` rounds; in round `k` this rank exchanges an
    /// empty control message with the peer `2^k` ahead of it and the peer
    /// `2^k` behind it (both indices modulo `size`). After every round has
    /// completed, every rank has a message-chain path to every other rank,
    /// which is what guarantees none can have returned before all arrived.
    /// Correct (not just for powers of two) because the `rounds` steps used
    /// (`1, 2, 4, ..., 2^(rounds-1)`) are all `< size` by construction, so
    /// they remain pairwise distinct modulo `size`, and each is its own
    /// unique round tag — no two rounds can ever address the same peer under
    /// the same tag.
    pub async fn barrier(&self) -> Result<(), ProcessError> {
        let size = self.size();
        if size <= 1 {
            return Ok(());
        }
        let endpoint = self.require_endpoint()?.clone();
        let rank = self.rank();
        let ctx = self.context.0;

        let mut rounds = 0usize;
        let mut cap = 1usize;
        while cap < size {
            cap <<= 1;
            rounds += 1;
        }

        for round in 0..rounds {
            let step = 1usize << round;
            let dst_local = collective::mod_add(rank, size, step);
            let src_local = collective::mod_sub(rank, size, step);
            let dst_global = self.global_rank(dst_local)?;
            let src_global = self.global_rank(src_local)?;
            let tag = collective::TAG_BARRIER_BASE + round as u64;
            endpoint
                .send_bytes(dst_global, ctx, tag, &[], SendOpts::default())
                .await?;
            endpoint.recv_bytes(src_global, ctx, tag).await?;
        }
        Ok(())
    }

    /// Split the communicator into sub-communicators based on color.
    ///
    /// Processes with the same `color` end up in the same sub-communicator;
    /// within a color, the new local rank order follows each member's
    /// *current* rank (used as the tie-breaking `key`, mirroring the common
    /// MPI idiom of passing your own rank as `key` when you don't need a
    /// different order).
    ///
    /// Implemented as `allgather(color, key)` — every rank learns every
    /// other rank's `(color, key)` — followed by purely local filtering,
    /// sorting and [`ContextId`] derivation: no further network round trips
    /// are needed once the `(color, key)` table is known. The child shares
    /// this communicator's `Arc<Endpoint>` (no new sockets); see
    /// `ContextId::derive` for why a *color-partitioned* split can safely
    /// hand every resulting sub-communicator the same derived context.
    pub async fn split(&self, color: usize) -> Result<Communicator, ProcessError> {
        // Derived *before* any fallible step below, and unconditionally on
        // every rank that calls `split` — including one that goes on to hit
        // an error and never returns a child communicator. If this were
        // deferred until just before constructing the `Ok(...)` value (as an
        // earlier revision did), a rank that errors out advances its
        // `split_seq` zero times while every rank that succeeds advances it
        // once, desynchronizing the counter between them for the *next*
        // call to `split`/`create_group` on this same parent — exactly the
        // cross-wiring [`ContextId::derive`]'s docs warn an endpoint-global
        // counter would cause, reintroduced at the per-communicator level if
        // the advance isn't unconditional.
        let child_context = self.derive_child_context();
        let size = self.size();
        if size <= 1 {
            // Nothing to negotiate: a lone rank forms its own singleton group.
            let new_group = ProcessGroup::new(vec![self.group.local_to_global_rank(0)?])?;
            let new_info = ProcessInfo::new(0, 1, self.info.addr, self.info.hostname.clone())?;
            return Ok(Communicator {
                info: Arc::new(new_info),
                group: Arc::new(new_group),
                addresses: Arc::clone(&self.addresses),
                endpoint: self.endpoint.clone(),
                context: child_context,
                split_seq: Arc::new(AtomicU64::new(0)),
            });
        }

        let key = self.rank() as u64;
        let (pairs, _counts) = collective::allgather_inner(
            &[(color as u64, key)],
            self,
            collective::TAG_SPLIT_ALLGATHER_BASE,
        )
        .await
        .map_err(|e| ProcessError::SplitFailed(e.to_string()))?;

        // `pairs[local_rank]` is the (color, key) that `local_rank` (within
        // `self`) contributed, since allgather concatenates in local-rank
        // order.
        let mut members: Vec<(usize, u64)> = pairs
            .iter()
            .enumerate()
            .filter(|(_, (c, _))| *c as usize == color)
            .map(|(local_rank, (_, k))| (local_rank, *k))
            .collect();
        // Break ties by original local rank for a fully deterministic order
        // even if two members pass the same key.
        members.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        let mut global_ranks = Vec::with_capacity(members.len());
        for (local_rank, _) in &members {
            global_ranks.push(self.group.local_to_global_rank(*local_rank)?);
        }
        let new_group = ProcessGroup::new(global_ranks)
            .map_err(|e| ProcessError::SplitFailed(e.to_string()))?;

        let my_global = self.group.local_to_global_rank(self.rank())?;
        let new_rank = new_group.global_to_local_rank(my_global)?;
        let new_info = ProcessInfo::new(
            new_rank,
            new_group.size(),
            self.info.addr,
            self.info.hostname.clone(),
        )
        .map_err(|e| ProcessError::SplitFailed(e.to_string()))?;

        let mut new_addresses = HashMap::new();
        for &g in &new_group.ranks {
            if let Some(addr) = self.addresses.get(&g) {
                new_addresses.insert(g, *addr);
            }
        }

        Ok(Communicator {
            info: Arc::new(new_info),
            group: Arc::new(new_group),
            addresses: Arc::new(new_addresses),
            endpoint: self.endpoint.clone(),
            context: child_context,
            split_seq: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Create a sub-communicator from an explicit, identical-on-every-member
    /// list of (this communicator's local) ranks.
    ///
    /// Unlike [`Self::split`], membership is supplied directly rather than
    /// negotiated via allgather — every member must call this with the
    /// exact same `ranks` (SPMD-style), which is how the resulting local
    /// rank order and derived [`ContextId`] end up agreeing across members
    /// without a network round trip. A rank not present in `ranks` gets
    /// [`ProcessError::InvalidRank`] rather than a null communicator.
    ///
    /// Every rank that calls this — including ones that error out below
    /// because they are not in `ranks` — advances `split_seq` by
    /// exactly one, for the same reason [`Self::split`] derives its child
    /// context unconditionally before any fallible step: see that method's
    /// comment.
    pub async fn create_group(&self, ranks: &[usize]) -> Result<Communicator, ProcessError> {
        let child_context = self.derive_child_context();
        for &rank in ranks {
            if rank >= self.size() {
                return Err(ProcessError::InvalidRank {
                    rank,
                    size: self.size(),
                });
            }
        }

        let mut global_ranks = Vec::with_capacity(ranks.len());
        for &r in ranks {
            global_ranks.push(self.group.local_to_global_rank(r)?);
        }
        let new_group = ProcessGroup::new(global_ranks)
            .map_err(|e| ProcessError::SplitFailed(e.to_string()))?;

        let my_global = self.group.local_to_global_rank(self.rank())?;
        let new_rank = new_group.global_to_local_rank(my_global)?;

        let new_info = ProcessInfo::new(
            new_rank,
            new_group.size(),
            self.info.addr,
            self.info.hostname.clone(),
        )
        .map_err(|e| ProcessError::SplitFailed(e.to_string()))?;

        let mut new_addresses = HashMap::new();
        for &rank in ranks {
            if let Ok(addr) = self.address(rank) {
                new_addresses.insert(rank, addr);
            }
        }

        Ok(Communicator {
            info: Arc::new(new_info),
            group: Arc::new(new_group),
            addresses: Arc::new(new_addresses),
            endpoint: self.endpoint.clone(),
            context: child_context,
            split_seq: Arc::new(AtomicU64::new(0)),
        })
    }
}

impl std::fmt::Debug for Communicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Communicator")
            .field("rank", &self.rank())
            .field("size", &self.size())
            .field("is_root", &self.is_root())
            .field("context", &self.context.0)
            .field("connected", &self.endpoint.is_some())
            .finish()
    }
}

/// The world communicator containing all processes
pub type WorldCommunicator = Communicator;

/// Global state for the distributed environment.
///
/// `RwLock<Option<...>>` (rather than the `OnceLock` this used to be) is
/// what lets [`finalize`] actually clear the slot: a `OnceLock` can only be
/// set once for the lifetime of the process, so a prior `finalize` could
/// never be followed by a fresh `init`. `std::sync::RwLock` (not
/// `tokio::sync::RwLock`) is deliberate: [`rank`]/[`size`] stay plain
/// synchronous functions this way, and every lock/unlock here is a quick
/// read-check-or-write with no `.await` in between, so a blocking lock is
/// never held across a suspension point.
static GLOBAL_WORLD: RwLock<Option<WorldCommunicator>> = RwLock::new(None);

/// Recover a poisoned lock's inner guard rather than panicking (COOLJAPAN
/// no-unwrap policy) — a panic in one task while holding the lock must not
/// permanently wedge every other task's access to `GLOBAL_WORLD`.
fn recover<T>(poisoned: std::sync::PoisonError<T>) -> T {
    poisoned.into_inner()
}

fn resolve_hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "localhost".to_string())
}

/// The historical "rank `r` listens on `127.0.0.1:5000+r`" convention
/// [`init`] has always used as its default addressing scheme.
fn convention_addr(rank: usize) -> SocketAddr {
    let port = 5000u16.saturating_add(u16::try_from(rank).unwrap_or(u16::MAX));
    SocketAddr::from(([127, 0, 0, 1], port))
}

/// Resolve [`init`]'s `(RendezvousMode, bind_addr)` pair from already-parsed
/// inputs. Factored out as a pure function (no env, no runtime) specifically
/// so the one case that is otherwise nearly impossible to unit-test —
/// `NUMRS2_MASTER_ADDR` set — has a test at all: [`GLOBAL_WORLD`] is one
/// process-global slot, so only a single in-process test could ever call
/// [`init`] for Master mode for real, and every other test in this module
/// already needs that same slot for its own `size == 1` runs.
///
/// For `size > 1` with no master address, every rank must independently
/// compute the *same* table without any exchange, so every entry other than
/// `rank`'s own follows [`convention_addr`] regardless of what that other
/// rank's own `NUMRS2_BIND_ADDR` might be set to — a pre-existing limitation
/// of this convention-only path, not one this function can resolve locally.
///
/// For `Some(master_addr)`, the default bind address is deliberately
/// **not** [`convention_addr`]: Master rendezvous exists precisely to
/// *discover* addresses through the exchange rather than agree on them by
/// convention, and every doc example in this crate sets
/// `NUMRS2_MASTER_ADDR=127.0.0.1:5000` — the exact address `convention_addr`
/// would then also hand rank 0 as its *own* data-endpoint bind address,
/// which rank 0 already has bound by the time it tries to also bind the
/// master exchange server there, failing with "address already in use". An
/// explicit `NUMRS2_BIND_ADDR` is still honored either way.
fn resolve_launch_mode(
    rank: usize,
    size: usize,
    master_addr: Option<SocketAddr>,
    explicit_bind_addr: Option<SocketAddr>,
) -> (RendezvousMode, SocketAddr) {
    if size <= 1 {
        let bind_addr = explicit_bind_addr.unwrap_or_else(|| convention_addr(rank));
        return (RendezvousMode::Static { addrs: Vec::new() }, bind_addr);
    }

    if let Some(addr) = master_addr {
        let bind_addr = explicit_bind_addr.unwrap_or(SocketAddr::from(([127, 0, 0, 1], 0)));
        return (RendezvousMode::Master { addr }, bind_addr);
    }

    let mut addrs = Vec::with_capacity(size);
    for r in 0..size {
        addrs.push(if r == rank {
            explicit_bind_addr.unwrap_or_else(|| convention_addr(rank))
        } else {
            convention_addr(r)
        });
    }
    let bind_addr = explicit_bind_addr.unwrap_or_else(|| convention_addr(rank));
    (RendezvousMode::Static { addrs }, bind_addr)
}

/// Initialize the distributed computing environment
///
/// This must be called before any other distributed operations.
/// Returns the world communicator containing all processes.
///
/// # Configuration
///
/// Configuration is read from environment variables (this is the historical
/// `NUMRS2_SIZE`/`NUMRS2_BIND_ADDR` contract predating [`super::bootstrap`];
/// see that module's docs for the newer `NUMRS2_WORLD_SIZE`/`NUMRS2_ADDRS`
/// contract used by [`super::bootstrap::bootstrap_from_env`]):
///
/// - `NUMRS2_RANK`: Process rank (default: 0)
/// - `NUMRS2_SIZE`: Total number of processes (default: 1)
/// - `NUMRS2_MASTER_ADDR`: Rendezvous server address (default: unset — see below)
/// - `NUMRS2_BIND_ADDR`: This process's bind address (default: "127.0.0.1:5000+rank")
///
/// When `NUMRS2_MASTER_ADDR` is set, rank 0 runs [`bootstrap`]'s master
/// exchange server and every rank (including 0) publishes/discovers
/// addresses through it. Otherwise, for `size > 1`, every rank independently
/// assumes the convention "rank `r` listens on `127.0.0.1:5000+r`" (using its
/// own resolved `NUMRS2_BIND_ADDR` for its own slot) — this preserves the
/// address convention this function has always used, now wired through the
/// real [`bootstrap::bootstrap`]/[`Endpoint::connect_mesh`] machinery instead
/// of fabricating a `Communicator` with no transport. `size == 1` binds a
/// solo endpoint with no mesh at all.
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
    {
        let guard = GLOBAL_WORLD.read().unwrap_or_else(recover);
        if guard.is_some() {
            return Err(ProcessError::AlreadyInitialized);
        }
    }

    let rank: usize = std::env::var("NUMRS2_RANK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let size: usize = std::env::var("NUMRS2_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    if size == 0 {
        return Err(ProcessError::InvalidSize(0));
    }
    if rank >= size {
        return Err(ProcessError::InvalidRank { rank, size });
    }

    let master_addr: Option<SocketAddr> = std::env::var("NUMRS2_MASTER_ADDR")
        .ok()
        .and_then(|s| s.parse().ok());
    let explicit_bind_addr: Option<SocketAddr> = std::env::var("NUMRS2_BIND_ADDR")
        .ok()
        .and_then(|s| s.parse().ok());

    let (mode, bind_addr) = resolve_launch_mode(rank, size, master_addr, explicit_bind_addr);

    let config = EndpointConfig::default();
    let endpoint = bootstrap::bootstrap(
        u32::try_from(rank).map_err(|_| ProcessError::InvalidRank { rank, size })?,
        u32::try_from(size).map_err(|_| ProcessError::InvalidSize(size))?,
        mode,
        bind_addr,
        config,
    )
    .await?;

    let world = Communicator::from_endpoint(Arc::new(endpoint))?;

    let mut guard = GLOBAL_WORLD.write().unwrap_or_else(recover);
    if guard.is_some() {
        return Err(ProcessError::AlreadyInitialized);
    }
    *guard = Some(world.clone());
    Ok(world)
}

/// Finalize the distributed computing environment
///
/// This should be called when all distributed operations are complete.
/// After calling finalize, no other distributed operations can be performed
/// until init() is called again — `GLOBAL_WORLD` is a real
/// `RwLock<Option<...>>` now, so unlike the old `OnceLock`-backed version of
/// this function, that "again" is real: the slot is genuinely cleared, and
/// the underlying [`Endpoint`] is genuinely shut down (every link closed,
/// every parked receive woken with an error) before that happens.
///
/// `GLOBAL_WORLD` is cleared **unconditionally** once finalize has confirmed
/// it was initialized, even if [`Endpoint::shutdown`] itself returns an
/// error: a shutdown failure is reported back to the caller, but must not
/// leave the slot permanently occupied by a communicator whose transport is
/// already torn down (or mid-teardown) — that would make a correctly-failed
/// finalize indistinguishable from a stuck `AlreadyInitialized` forever.
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
pub async fn finalize(world: WorldCommunicator) -> Result<(), ProcessError> {
    {
        let guard = GLOBAL_WORLD.read().unwrap_or_else(recover);
        if guard.is_none() {
            return Err(ProcessError::NotInitialized);
        }
    }

    let shutdown_result = match world.endpoint.as_ref() {
        Some(endpoint) => endpoint.shutdown().await,
        None => Ok(()),
    };

    let mut guard = GLOBAL_WORLD.write().unwrap_or_else(recover);
    *guard = None;
    drop(guard);

    shutdown_result.map_err(ProcessError::from)
}

/// Get the rank of the current process in the world communicator
///
/// Convenience function that returns the rank without needing a communicator reference.
/// Requires that init() has been called.
pub fn rank() -> Result<usize, ProcessError> {
    let guard = GLOBAL_WORLD.read().unwrap_or_else(recover);
    guard
        .as_ref()
        .map(|w| w.rank())
        .ok_or(ProcessError::NotInitialized)
}

/// Get the size of the world communicator
///
/// Convenience function that returns the size without needing a communicator reference.
/// Requires that init() has been called.
pub fn size() -> Result<usize, ProcessError> {
    let guard = GLOBAL_WORLD.read().unwrap_or_else(recover);
    guard
        .as_ref()
        .map(|w| w.size())
        .ok_or(ProcessError::NotInitialized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::collective::{allreduce, ReduceOp};
    use crate::distributed::net::SendOpts as NetSendOpts;
    use crate::distributed::testing::{ClusterNode, LocalCluster};
    use serial_test::serial;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    fn short_timeout_config() -> EndpointConfig {
        EndpointConfig {
            recv_timeout: Duration::from_secs(2),
            ..EndpointConfig::default()
        }
    }

    // -------------------------------------------------------------
    // resolve_launch_mode: pure, so every branch is directly testable
    // without a runtime or GLOBAL_WORLD (which only one in-process test
    // could ever legitimately claim for a real `init()` call).
    // -------------------------------------------------------------

    #[test]
    fn resolve_launch_mode_size_one_defaults_to_the_convention_port() {
        let (mode, bind_addr) = resolve_launch_mode(0, 1, None, None);
        assert!(matches!(mode, RendezvousMode::Static { ref addrs } if addrs.is_empty()));
        assert_eq!(bind_addr, "127.0.0.1:5000".parse().expect("valid"));
    }

    #[test]
    fn resolve_launch_mode_size_one_honors_explicit_bind_addr() {
        let explicit: SocketAddr = "127.0.0.1:9999".parse().expect("valid");
        let (_, bind_addr) = resolve_launch_mode(0, 1, None, Some(explicit));
        assert_eq!(bind_addr, explicit);
    }

    /// Regression test for the bug an earlier revision of this function had:
    /// with `NUMRS2_MASTER_ADDR` set and no explicit `NUMRS2_BIND_ADDR`, the
    /// data-endpoint bind address must be an OS-assigned ephemeral port
    /// (`:0`), *not* the `5000 + rank` convention — reusing that convention
    /// makes rank 0's own data-endpoint bind address collide with the
    /// `NUMRS2_MASTER_ADDR=127.0.0.1:5000` this crate's own examples default
    /// to, since rank 0 binds both. See [`resolve_launch_mode`]'s docs.
    #[test]
    fn resolve_launch_mode_master_mode_defaults_to_an_ephemeral_bind_addr_not_the_convention() {
        let master: SocketAddr = "127.0.0.1:5000".parse().expect("valid");
        let (mode, bind_addr) = resolve_launch_mode(0, 4, Some(master), None);
        assert!(matches!(mode, RendezvousMode::Master { addr } if addr == master));
        assert_eq!(
            bind_addr.port(),
            0,
            "must not collide with NUMRS2_MASTER_ADDR"
        );
        assert_ne!(
            bind_addr, master,
            "rank 0's own data endpoint must never default to the master's own exchange address"
        );
    }

    #[test]
    fn resolve_launch_mode_master_mode_still_honors_an_explicit_bind_addr() {
        let master: SocketAddr = "127.0.0.1:5000".parse().expect("valid");
        let explicit: SocketAddr = "127.0.0.1:6100".parse().expect("valid");
        let (_, bind_addr) = resolve_launch_mode(1, 4, Some(master), Some(explicit));
        assert_eq!(bind_addr, explicit);
    }

    #[test]
    fn resolve_launch_mode_convention_table_uses_explicit_bind_only_for_own_rank() {
        let explicit: SocketAddr = "10.0.0.5:7000".parse().expect("valid");
        let (mode, bind_addr) = resolve_launch_mode(1, 3, None, Some(explicit));
        assert_eq!(bind_addr, explicit);
        match mode {
            RendezvousMode::Static { addrs } => {
                assert_eq!(addrs.len(), 3);
                assert_eq!(addrs[0], "127.0.0.1:5000".parse().expect("valid"));
                assert_eq!(
                    addrs[1], explicit,
                    "rank 1's own slot uses its explicit bind addr"
                );
                assert_eq!(addrs[2], "127.0.0.1:5002".parse().expect("valid"));
            }
            other => panic!("expected Static, got {other:?}"),
        }
    }

    #[test]
    fn resolve_launch_mode_convention_table_with_no_explicit_bind() {
        let (mode, bind_addr) = resolve_launch_mode(2, 4, None, None);
        assert_eq!(bind_addr, "127.0.0.1:5002".parse().expect("valid"));
        match mode {
            RendezvousMode::Static { addrs } => {
                let expected: Vec<SocketAddr> = (0..4)
                    .map(|r| format!("127.0.0.1:{}", 5000 + r).parse().expect("valid"))
                    .collect();
                assert_eq!(addrs, expected);
            }
            other => panic!("expected Static, got {other:?}"),
        }
    }

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
        assert!(comm.endpoint().is_none());
        assert!(comm.require_endpoint().is_err());
    }

    #[tokio::test]
    async fn offline_communicator_barrier_is_a_size_one_noop() {
        let addr: SocketAddr = "127.0.0.1:5000".parse().expect("valid addr");
        let info = ProcessInfo::new(0, 1, addr, "localhost".to_string()).expect("valid info");
        let group = ProcessGroup::new(vec![0]).expect("valid group");
        let comm = Communicator::new(info, group, HashMap::new()).expect("valid communicator");
        // Must succeed without ever touching the (absent) endpoint.
        comm.barrier().await.expect("size-1 barrier is a no-op");
        let child = comm.split(0).await.expect("size-1 split is a no-op");
        assert_eq!(child.size(), 1);
        assert_eq!(child.rank(), 0);
    }

    #[test]
    fn context_id_derive_is_deterministic_and_avoids_reserved_value() {
        let parent = ContextId::WORLD;
        let a = ContextId::derive(parent, 0);
        let b = ContextId::derive(parent, 0);
        let c = ContextId::derive(parent, 1);
        assert_eq!(a, b, "same (parent, seq) must derive the same context");
        assert_ne!(a, c, "different seq must derive a different context");
        assert_ne!(a.0, u64::MAX, "derived context must never be CTX_CONTROL");
        assert_ne!(c.0, u64::MAX, "derived context must never be CTX_CONTROL");
    }

    #[test]
    fn context_id_derive_lets_sibling_colors_reuse_one_context_safely() {
        // Both colors produced by the *same* split call derive from the same
        // (parent, seq) pair — this is intentional (see ContextId::derive's
        // docs): colors partition the parent's ranks, so no rank is ever a
        // member of two same-context groups simultaneously.
        let parent = ContextId::WORLD;
        let color0_ctx = ContextId::derive(parent, 0);
        let color1_ctx = ContextId::derive(parent, 0);
        assert_eq!(color0_ctx, color1_ctx);
    }

    /// `init -> finalize -> init` at `size == 1` must succeed both times:
    /// `GLOBAL_WORLD` genuinely clears (this used to be impossible with the
    /// `OnceLock`-backed global), and the underlying endpoint's `shutdown`
    /// actually runs before that. Env vars and `GLOBAL_WORLD` are both
    /// process-global, hence `#[serial]` (matching `bootstrap.rs`'s own
    /// convention for its env-var tests). Binds an OS-assigned ephemeral
    /// port (`NUMRS2_BIND_ADDR=127.0.0.1:0`) rather than the default
    /// `127.0.0.1:5000`: `Endpoint::shutdown` aborts its accept task
    /// asynchronously (see `net::endpoint::Endpoint::shutdown`, not owned by
    /// this lane) with no guarantee the OS has released the port by the time
    /// `finalize` returns, so re-binding the exact same fixed port
    /// immediately afterward is not something this test should depend on.
    #[tokio::test]
    #[serial]
    async fn init_finalize_init_round_trip_at_size_one() {
        for var in [
            "NUMRS2_RANK",
            "NUMRS2_SIZE",
            "NUMRS2_MASTER_ADDR",
            "NUMRS2_BIND_ADDR",
        ] {
            std::env::remove_var(var);
        }
        std::env::set_var("NUMRS2_SIZE", "1");
        std::env::set_var("NUMRS2_BIND_ADDR", "127.0.0.1:0");

        let world1 = init().await.expect("first init should succeed");
        assert_eq!(world1.rank(), 0);
        assert_eq!(world1.size(), 1);
        assert!(rank().is_ok());

        let second_attempt = init().await;
        assert!(matches!(
            second_attempt,
            Err(ProcessError::AlreadyInitialized)
        ));

        finalize(world1).await.expect("finalize should succeed");
        assert!(matches!(rank(), Err(ProcessError::NotInitialized)));

        let world2 = init().await.expect("re-init after finalize should succeed");
        assert_eq!(world2.rank(), 0);
        finalize(world2)
            .await
            .expect("second finalize should succeed");

        std::env::remove_var("NUMRS2_SIZE");
        std::env::remove_var("NUMRS2_BIND_ADDR");
    }

    /// Every rank stamps a shared phase counter immediately before calling
    /// `barrier`, staggered so ranks arrive at very different times; by the
    /// time *any* rank observes the counter right after its own `barrier`
    /// returns, every rank must already have stamped it — proving the real
    /// dissemination barrier actually withholds every rank until all have
    /// arrived, for every world size the task calls for.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn barrier_releases_all_ranks_together() {
        for world_size in 1..=4u32 {
            let phase = Arc::new(AtomicUsize::new(0));
            let phase_for_body = Arc::clone(&phase);
            let cfg = short_timeout_config();
            let results = LocalCluster::run_connected_with(
                world_size,
                cfg,
                Duration::from_secs(15),
                move |node: ClusterNode| {
                    let phase = Arc::clone(&phase_for_body);
                    async move {
                        let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                        tokio::time::sleep(Duration::from_millis(5 * comm.rank() as u64)).await;
                        phase.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        comm.barrier().await?;
                        Ok(phase.load(std::sync::atomic::Ordering::SeqCst))
                    }
                },
            )
            .await
            .unwrap_or_else(|e| panic!("barrier run for world_size={world_size} failed: {e}"));

            for seen in results {
                assert_eq!(
                    seen, world_size as usize,
                    "a rank exited the barrier before every rank had arrived (world_size={world_size})"
                );
            }
        }
    }

    /// `split` by parity forms exactly two sub-communicators of the correct
    /// size/rank, and a real collective run *inside* each child communicator
    /// (proving it independently shares the parent's endpoint under its own
    /// context) produces the expected result regardless of which color it
    /// is.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn split_by_parity_forms_two_working_subcommunicators() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            4,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                let color = comm.rank() % 2;
                let sub = comm.split(color).await?;
                let local = vec![(sub.rank() + 1) as f64];
                let total = allreduce(&local, ReduceOp::Sum, &sub).await?;
                Ok((color, sub.size(), sub.rank(), total))
            },
        )
        .await
        .expect("split run");

        assert_eq!(results.len(), 4);
        let mut colors_seen: Vec<usize> = Vec::new();
        for (color, size, rank, total) in &results {
            assert_eq!(*size, 2, "each parity group should have exactly 2 members");
            assert!(*rank < 2, "new local rank must be within the child's size");
            // Child has 2 members with ranks {0,1}; sum of (rank+1) = 1+2 = 3
            // regardless of which color this child is.
            assert_eq!(total, &vec![3.0]);
            colors_seen.push(*color);
        }
        colors_seen.sort_unstable();
        assert_eq!(colors_seen, vec![0, 0, 1, 1]);
    }

    /// Regression test: `create_group(&[0, 1])` succeeds for ranks 0 and 1
    /// but fails for ranks 2 and 3 (not members). If the failing ranks'
    /// `split_seq` counter did not still advance by exactly one on that
    /// call (an earlier revision of `split`/`create_group` only derived the
    /// child context, and therefore only advanced the counter, on the
    /// success path), a *following* `split` on the same parent would derive
    /// mismatched child `ContextId`s for ranks that land in the same color
    /// group (e.g. rank 0 at `split_seq == 1` versus rank 2 still at
    /// `split_seq == 0`), and the collective run inside that child below
    /// would hang until `RecvTimeout` instead of completing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_group_failure_does_not_desync_the_parents_split_seq_counter() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            4,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(Arc::new(node.endpoint))?;
                // Every rank calls this; only 0 and 1 are members of [0, 1].
                let _ = comm.create_group(&[0, 1]).await;

                let color = comm.rank() % 2;
                let sub = comm.split(color).await?;
                let local = vec![(sub.rank() + 1) as i64];
                allreduce(&local, ReduceOp::Sum, &sub)
                    .await
                    .map_err(NetError::from)
            },
        )
        .await
        .expect("a split_seq desync would hang this until RecvTimeout, not fail cleanly");

        for total in results {
            assert_eq!(total, vec![3]);
        }
    }

    // Exercised indirectly by the collective tests in `super::super::collective`,
    // which build communicators via `Communicator::from_endpoint` the same
    // way `super::testing::LocalCluster::run_connected` harnesses do; kept
    // here too as a direct smoke test that `SendOpts`/`Endpoint` remain
    // reachable from this module's own re-exports.
    #[test]
    fn net_send_opts_default_is_reachable_from_process_module() {
        assert_eq!(NetSendOpts::default(), NetSendOpts { compress: false });
    }
}
