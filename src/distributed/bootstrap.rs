//! Environment-variable bootstrap contract for launching a distributed run.
//!
//! A distributed NumRS2 run is configured entirely through environment
//! variables (no config file, no CLI flags), so it launches the same way
//! under a hand-rolled shell loop, `mpirun`-style launchers, or a scheduler
//! that just sets env vars per task. Every variable has a `NUMRS2_`-prefixed
//! canonical name and a shorter `NUMRS_`-prefixed alias; when both are set,
//! `NUMRS2_*` wins.
//!
//! | Variable             | Alias               | Meaning                                             | Default |
//! |----------------------|----------------------|------------------------------------------------------|---------|
//! | `NUMRS2_RANK`        | `NUMRS_RANK`         | This process's rank, `0..world_size`                  | `0`     |
//! | `NUMRS2_WORLD_SIZE`  | `NUMRS_WORLD_SIZE`   | Total number of ranks                                 | `1`     |
//! | `NUMRS2_ADDRS`       | `NUMRS_ADDRS`        | Comma-separated `host:port` list, one per rank, in rank order | unset (empty) |
//! | `NUMRS2_MASTER_ADDR` | `NUMRS_MASTER_ADDR`  | Rendezvous/coordinator address                        | unset (`None`) |
//!
//! `world_size == 1` (see [`SINGLE_PROCESS_WORLD_SIZE`] and
//! [`BootstrapConfig::is_single_process`]) means there is no one else to
//! talk to: callers should short-circuit collectives and point-to-point
//! operations to local no-ops/identity rather than standing up any
//! networking at all.
//!
//! # Relationship to `process::init()`
//!
//! [`super::process::init`] predates this module and reads its own,
//! narrower env contract (`NUMRS2_RANK` / `NUMRS2_SIZE` / `NUMRS2_MASTER_ADDR`
//! / `NUMRS2_BIND_ADDR` — note `NUMRS2_SIZE`, not `NUMRS2_WORLD_SIZE`) with
//! no `NUMRS_*` aliases. The `examples/distributed_basics.rs` and
//! `examples/distributed_computing.rs` launch instructions document that
//! old name. Whichever lane replaces `process.rs`'s transport with `net::`
//! needs to either migrate those call sites/examples to
//! `NUMRS2_WORLD_SIZE`, or have [`BootstrapConfig::from_env`] additionally
//! accept `NUMRS2_SIZE` as one more alias — this module does not do that
//! today, so a launch script using only the old name currently gets
//! `world_size = 1` here rather than an error.
//!
//! # Rendezvous: how ranks find each other
//!
//! Reading the environment only tells a process *who it is*. Turning that
//! into a working mesh is [`bootstrap`], in three steps:
//!
//! 1. **bind** — stand up this rank's [`Endpoint`] on its listening socket
//!    (`127.0.0.1:0` for an OS-assigned port unless the mode dictates one).
//! 2. **exchange** — learn every rank's bound address, by one of three
//!    [`RendezvousMode`]s: [`Static`](RendezvousMode::Static) (the table was
//!    handed to us in `NUMRS2_ADDRS`), [`Master`](RendezvousMode::Master)
//!    (rank 0 runs a tiny exchange server every rank publishes to and reads
//!    the full table back from), or
//!    [`InProcess`](RendezvousMode::InProcess) (an in-memory table, for
//!    [`super::testing::LocalCluster`]).
//! 3. **connect_mesh** — [`Endpoint::connect_mesh`], with the pair rule
//!    *`j` dials `i` whenever `i < j`* and a HELLO frame carrying the
//!    dialer's rank.
//!
//! `world_size == 1` short-circuits after step 1: there is nobody to
//! exchange with and nothing to dial, and self-sends never leave the
//! mailbox.
//!
//! Note that `InProcess` is an in-memory *address table*, not an in-memory
//! transport: those ranks still talk over real loopback TCP, so a test run
//! under [`super::testing::LocalCluster`] exercises the same framing,
//! queueing and connection code a multi-host run does.

use super::net::endpoint::Endpoint;
use super::net::frame::{FrameHeader, CTX_CONTROL, TAG_BOOTSTRAP_PUBLISH, TAG_BOOTSTRAP_TABLE};
use super::net::link::{connect_with_retry, read_frame, write_frame};
use super::net::{EndpointConfig, NetError};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;

/// A `world_size` of exactly this value means single-process (no peers).
pub const SINGLE_PROCESS_WORLD_SIZE: u32 = 1;

/// This process's resolved rank and world size, plus whatever addressing
/// information the environment provided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapConfig {
    /// This process's rank, `0..world_size`.
    pub rank: u32,
    /// Total number of ranks.
    pub world_size: u32,
    /// Per-rank addresses from `NUMRS2_ADDRS`/`NUMRS_ADDRS`, in rank order.
    /// Empty when that variable is unset.
    pub addrs: Vec<SocketAddr>,
    /// Rendezvous/coordinator address from
    /// `NUMRS2_MASTER_ADDR`/`NUMRS_MASTER_ADDR`, if set.
    pub master_addr: Option<SocketAddr>,
}

impl BootstrapConfig {
    /// Read the full bootstrap contract from the process environment. See
    /// the module docs for variable names, aliases, and defaults.
    pub fn from_env() -> Result<Self, NetError> {
        let rank = env_u32("NUMRS2_RANK", "NUMRS_RANK", 0)?;
        let world_size = env_u32(
            "NUMRS2_WORLD_SIZE",
            "NUMRS_WORLD_SIZE",
            SINGLE_PROCESS_WORLD_SIZE,
        )?;
        let addrs = env_addr_list("NUMRS2_ADDRS", "NUMRS_ADDRS")?;
        let master_addr = env_addr_opt("NUMRS2_MASTER_ADDR", "NUMRS_MASTER_ADDR")?;

        if world_size == 0 {
            return Err(NetError::Bootstrap(
                "world_size must be >= 1, got 0".to_string(),
            ));
        }
        if rank >= world_size {
            return Err(NetError::InvalidRank {
                rank,
                size: world_size,
            });
        }
        if !addrs.is_empty() && addrs.len() != world_size as usize {
            return Err(NetError::Bootstrap(format!(
                "NUMRS2_ADDRS lists {} address(es) but world_size is {world_size}",
                addrs.len(),
            )));
        }

        Ok(Self {
            rank,
            world_size,
            addrs,
            master_addr,
        })
    }

    /// True for a single-process "world" (`world_size == 1`): collectives
    /// and point-to-point calls should short-circuit rather than open any
    /// connection.
    pub fn is_single_process(&self) -> bool {
        self.world_size == SINGLE_PROCESS_WORLD_SIZE
    }
}

/// Read `primary`, falling back to `alias`. `primary` wins when both are set.
fn env_str(primary: &str, alias: &str) -> Option<String> {
    std::env::var(primary)
        .ok()
        .or_else(|| std::env::var(alias).ok())
}

fn env_u32(primary: &str, alias: &str, default: u32) -> Result<u32, NetError> {
    match env_str(primary, alias) {
        None => Ok(default),
        Some(raw) => raw.trim().parse::<u32>().map_err(|e| {
            NetError::Bootstrap(format!(
                "{primary} (or {alias}) = {raw:?} is not a u32: {e}"
            ))
        }),
    }
}

fn env_addr_opt(primary: &str, alias: &str) -> Result<Option<SocketAddr>, NetError> {
    match env_str(primary, alias) {
        None => Ok(None),
        Some(raw) => raw.trim().parse::<SocketAddr>().map(Some).map_err(|e| {
            NetError::Bootstrap(format!(
                "{primary} (or {alias}) = {raw:?} is not a socket address: {e}"
            ))
        }),
    }
}

fn env_addr_list(primary: &str, alias: &str) -> Result<Vec<SocketAddr>, NetError> {
    match env_str(primary, alias) {
        None => Ok(Vec::new()),
        Some(raw) => raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<SocketAddr>().map_err(|e| {
                    NetError::Bootstrap(format!(
                        "{primary} (or {alias}): {s:?} is not a socket address: {e}"
                    ))
                })
            })
            .collect(),
    }
}

// ===================================================================
// Rendezvous
// ===================================================================

/// Shared, in-memory rank → address table.
///
/// This is the `InProcess` rendezvous back-end: the only thing it replaces
/// is *address discovery*. Ranks that find each other through it still
/// connect over real loopback TCP, so [`super::testing::LocalCluster`]
/// exercises the production framing, queueing and connection paths rather
/// than a shortcut around them.
#[derive(Clone)]
pub struct InProcessRegistry {
    inner: Arc<InProcessInner>,
}

struct InProcessInner {
    world_size: u32,
    addrs: Mutex<BTreeMap<u32, SocketAddr>>,
    notify: Notify,
}

impl std::fmt::Debug for InProcessRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let registered = self.locked().len();
        f.debug_struct("InProcessRegistry")
            .field("world_size", &self.inner.world_size)
            .field("registered", &registered)
            .finish()
    }
}

impl InProcessRegistry {
    /// A table expecting `world_size` ranks to publish.
    pub fn new(world_size: u32) -> Self {
        Self {
            inner: Arc::new(InProcessInner {
                world_size,
                addrs: Mutex::new(BTreeMap::new()),
                notify: Notify::new(),
            }),
        }
    }

    /// How many ranks this table waits for in [`Self::all_addrs`].
    pub fn world_size(&self) -> u32 {
        self.inner.world_size
    }

    fn locked(&self) -> std::sync::MutexGuard<'_, BTreeMap<u32, SocketAddr>> {
        self.inner
            .addrs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Publish this rank's bound address. Idempotent: registering the same
    /// rank again replaces the previous address.
    pub fn register(&self, rank: u32, addr: SocketAddr) {
        let mut guard = self.locked();
        guard.insert(rank, addr);
        drop(guard);
        self.inner.notify.notify_waiters();
    }

    /// Wait for and return `rank`'s bound address.
    pub async fn lookup(&self, rank: u32) -> SocketAddr {
        loop {
            {
                if let Some(addr) = self.locked().get(&rank) {
                    return *addr;
                }
            }
            self.inner.notify.notified().await;
        }
    }

    /// Wait for every rank in `0..world_size` to register, then return every
    /// address in rank order.
    pub async fn all_addrs(&self) -> Vec<SocketAddr> {
        loop {
            {
                let guard = self.locked();
                if guard.len() as u32 >= self.inner.world_size {
                    return guard.values().copied().collect();
                }
            }
            self.inner.notify.notified().await;
        }
    }
}

/// How the ranks of a run discover each other's bound addresses.
#[derive(Debug, Clone)]
pub enum RendezvousMode {
    /// The full table was supplied up front (`NUMRS2_ADDRS`). No exchange
    /// traffic at all: every rank already knows every address, and binds the
    /// one listed for its own rank.
    Static {
        /// Every rank's address, in rank order.
        addrs: Vec<SocketAddr>,
    },
    /// Rank 0 runs a tiny exchange server at this address. Every rank
    /// (including rank 0) connects to it, publishes its own bound data
    /// address, and receives the completed table back.
    Master {
        /// Where rank 0's exchange server listens.
        addr: SocketAddr,
    },
    /// An in-memory table shared by ranks in one process
    /// ([`super::testing::LocalCluster`]).
    InProcess(InProcessRegistry),
}

impl BootstrapConfig {
    /// The rendezvous mode this environment implies.
    ///
    /// A `NUMRS2_ADDRS` table wins (it needs no exchange at all); otherwise a
    /// `NUMRS2_MASTER_ADDR` selects the master exchange. A single-process
    /// world needs neither, and is reported as an empty
    /// [`RendezvousMode::Static`] table of one.
    pub fn rendezvous_mode(&self) -> Result<RendezvousMode, NetError> {
        if !self.addrs.is_empty() {
            return Ok(RendezvousMode::Static {
                addrs: self.addrs.clone(),
            });
        }
        if let Some(addr) = self.master_addr {
            return Ok(RendezvousMode::Master { addr });
        }
        if self.is_single_process() {
            return Ok(RendezvousMode::Static { addrs: Vec::new() });
        }
        Err(NetError::Bootstrap(format!(
            "world_size is {} but neither NUMRS2_ADDRS nor NUMRS2_MASTER_ADDR is set",
            self.world_size
        )))
    }
}

/// Bind, exchange addresses, and connect the full mesh — the whole launch
/// sequence for one rank.
///
/// `bind_addr` is this rank's listening address; pass `127.0.0.1:0` (or any
/// port-0 address) to let the OS choose, which is what
/// [`RendezvousMode::Master`] and [`RendezvousMode::InProcess`] expect since
/// the real address is discovered rather than pre-agreed.
/// [`RendezvousMode::Static`] instead binds the address the table lists for
/// this rank, ignoring `bind_addr`.
///
/// Returns a fully connected [`Endpoint`]: every peer link is up before this
/// resolves, so an immediately following `send` can never find one missing.
pub async fn bootstrap(
    rank: u32,
    world_size: u32,
    mode: RendezvousMode,
    bind_addr: SocketAddr,
    config: EndpointConfig,
) -> Result<Endpoint, NetError> {
    if world_size == 0 || rank >= world_size {
        return Err(NetError::InvalidRank {
            rank,
            size: world_size,
        });
    }

    // Step 1: bind. A static table pins this rank's address; every other
    // mode discovers whatever the OS handed us.
    let bind_addr = match &mode {
        RendezvousMode::Static { addrs } if !addrs.is_empty() => {
            *addrs.get(rank as usize).ok_or_else(|| {
                NetError::Bootstrap(format!(
                    "static address table has {} entries, no slot for rank {rank}",
                    addrs.len()
                ))
            })?
        }
        _ => bind_addr,
    };
    let endpoint = Endpoint::bind(bind_addr, rank, world_size, config).await?;

    // A world of one has nobody to exchange with and nothing to dial.
    if world_size == SINGLE_PROCESS_WORLD_SIZE {
        return Ok(endpoint);
    }

    // Step 2: exchange.
    let addrs = exchange(rank, world_size, &mode, &endpoint).await?;
    if addrs.len() != world_size as usize {
        return Err(NetError::Bootstrap(format!(
            "rendezvous produced {} address(es) for a world of {world_size}",
            addrs.len()
        )));
    }

    // Step 3: connect the mesh.
    endpoint.connect_mesh(&addrs).await?;
    Ok(endpoint)
}

/// Convenience wrapper: read the environment, then [`bootstrap`] with it.
pub async fn bootstrap_from_env(config: EndpointConfig) -> Result<Endpoint, NetError> {
    let cfg = BootstrapConfig::from_env()?;
    let mode = cfg.rendezvous_mode()?;
    let ephemeral = SocketAddr::from(([127, 0, 0, 1], 0));
    bootstrap(cfg.rank, cfg.world_size, mode, ephemeral, config).await
}

/// Step 2 of [`bootstrap`]: resolve the full rank-ordered address table.
async fn exchange(
    rank: u32,
    world_size: u32,
    mode: &RendezvousMode,
    endpoint: &Endpoint,
) -> Result<Vec<SocketAddr>, NetError> {
    match mode {
        RendezvousMode::Static { addrs } => Ok(addrs.clone()),
        RendezvousMode::InProcess(registry) => {
            registry.register(rank, endpoint.local_addr());
            Ok(registry.all_addrs().await)
        }
        RendezvousMode::Master { addr } => master_exchange(rank, world_size, *addr, endpoint).await,
    }
}

/// `Master` rendezvous: rank 0 serves the exchange while also taking part in
/// it as an ordinary client.
async fn master_exchange(
    rank: u32,
    world_size: u32,
    master_addr: SocketAddr,
    endpoint: &Endpoint,
) -> Result<Vec<SocketAddr>, NetError> {
    let config = endpoint.config().clone();
    let local_addr = endpoint.local_addr();

    if rank != 0 {
        return master_client(master_addr, rank, local_addr, &config).await;
    }

    let listener = TcpListener::bind(master_addr)
        .await
        .map_err(|e| NetError::ConnectFailed {
            addr: master_addr.to_string(),
            msg: format!("master exchange server could not bind: {e}"),
        })?;
    // Serve and participate on the same task: `join!` interleaves the accept
    // loop with rank 0's own publish/receive, so neither starves the other.
    let (served, client) = tokio::join!(
        serve_master_exchange(listener, world_size, &config),
        master_client(master_addr, rank, local_addr, &config)
    );
    served?;
    client
}

/// Rank 0's exchange server: collect one address per rank, then hand every
/// rank the completed table.
///
/// Speaks the ordinary [`FrameHeader`] codec on a throwaway socket rather
/// than inventing a second parser.
async fn serve_master_exchange(
    listener: TcpListener,
    world_size: u32,
    config: &EndpointConfig,
) -> Result<(), NetError> {
    let deadline = tokio::time::Instant::now() + config.connect_timeout;
    let mut addrs: BTreeMap<u32, SocketAddr> = BTreeMap::new();
    let mut clients: BTreeMap<u32, TcpStream> = BTreeMap::new();

    while (addrs.len() as u32) < world_size {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(NetError::Bootstrap(format!(
                "master exchange timed out with {} of {world_size} rank(s) checked in",
                addrs.len()
            )));
        }
        let accepted = tokio::time::timeout(remaining, listener.accept())
            .await
            .map_err(|_| {
                NetError::Bootstrap(format!(
                    "master exchange timed out with {} of {world_size} rank(s) checked in",
                    addrs.len()
                ))
            })?;
        let (mut stream, _peer) = accepted.map_err(|e| NetError::Io(e.to_string()))?;

        let (header, payload) = match tokio::time::timeout(
            config.connect_timeout,
            read_frame(&mut stream, config.max_frame),
        )
        .await
        {
            // A client that connects and then misbehaves must not take the
            // whole rendezvous down: drop it and keep serving.
            Err(_) | Ok(Err(_)) => continue,
            Ok(Ok(frame)) => frame,
        };
        if header.ctx != CTX_CONTROL || header.tag != TAG_BOOTSTRAP_PUBLISH {
            continue;
        }
        if header.src >= world_size {
            continue;
        }
        let Ok(text) = std::str::from_utf8(&payload) else {
            continue;
        };
        let Ok(addr) = text.trim().parse::<SocketAddr>() else {
            continue;
        };
        addrs.insert(header.src, addr);
        clients.insert(header.src, stream);
    }

    let table = encode_table(addrs.values().copied());
    let payload = table.as_bytes();
    let len = u32::try_from(payload.len()).map_err(|_| NetError::FrameTooLarge {
        size: payload.len(),
        max: config.max_frame,
    })?;
    for (rank, mut stream) in clients {
        let header = FrameHeader::new(0, rank, CTX_CONTROL, TAG_BOOTSTRAP_TABLE, 0, len, len, 0)?;
        write_frame(&mut stream, &header, payload).await?;
    }
    Ok(())
}

/// One rank's side of the master exchange: publish, then read the table back.
async fn master_client(
    master_addr: SocketAddr,
    rank: u32,
    local_addr: SocketAddr,
    config: &EndpointConfig,
) -> Result<Vec<SocketAddr>, NetError> {
    let mut stream = connect_with_retry(master_addr, config.connect_timeout).await?;

    let advertised = local_addr.to_string();
    let payload = advertised.as_bytes();
    let len = u32::try_from(payload.len()).map_err(|_| NetError::FrameTooLarge {
        size: payload.len(),
        max: config.max_frame,
    })?;
    let header = FrameHeader::new(rank, 0, CTX_CONTROL, TAG_BOOTSTRAP_PUBLISH, 0, len, len, 0)?;
    write_frame(&mut stream, &header, payload).await?;

    let (header, payload) = tokio::time::timeout(
        config.connect_timeout,
        read_frame(&mut stream, config.max_frame),
    )
    .await
    .map_err(|_| {
        NetError::Bootstrap(format!(
            "rank {rank} timed out waiting for the address table from {master_addr}"
        ))
    })??;
    if header.ctx != CTX_CONTROL || header.tag != TAG_BOOTSTRAP_TABLE {
        return Err(NetError::Bootstrap(format!(
            "expected the address table from the master, got ctx {} tag {}",
            header.ctx, header.tag
        )));
    }
    decode_table(&payload)
}

/// Render a rank-ordered address table as the comma-separated UTF-8 list the
/// master exchange sends on the wire.
fn encode_table<I: IntoIterator<Item = SocketAddr>>(addrs: I) -> String {
    addrs
        .into_iter()
        .map(|a| a.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Parse the comma-separated table produced by [`encode_table`].
fn decode_table(payload: &[u8]) -> Result<Vec<SocketAddr>, NetError> {
    let text = std::str::from_utf8(payload)
        .map_err(|e| NetError::Bootstrap(format!("address table is not UTF-8: {e}")))?;
    text.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<SocketAddr>().map_err(|e| {
                NetError::Bootstrap(format!("address table entry {s:?} is not an address: {e}"))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::net::SendOpts;
    use serial_test::serial;
    use std::time::Duration;

    const ALL_VARS: &[&str] = &[
        "NUMRS2_RANK",
        "NUMRS_RANK",
        "NUMRS2_WORLD_SIZE",
        "NUMRS_WORLD_SIZE",
        "NUMRS2_ADDRS",
        "NUMRS_ADDRS",
        "NUMRS2_MASTER_ADDR",
        "NUMRS_MASTER_ADDR",
    ];

    /// Removes every bootstrap env var. Tests in this module run
    /// `#[serial]` (env vars are process-global) and must clean up both
    /// before and after so a failure doesn't leak state into the next test.
    fn clear_env() {
        for var in ALL_VARS {
            std::env::remove_var(var);
        }
    }

    #[test]
    #[serial]
    fn defaults_to_single_process() {
        clear_env();
        let cfg = BootstrapConfig::from_env().expect("defaults are valid");
        assert_eq!(cfg.rank, 0);
        assert_eq!(cfg.world_size, SINGLE_PROCESS_WORLD_SIZE);
        assert!(cfg.is_single_process());
        assert!(cfg.addrs.is_empty());
        assert!(cfg.master_addr.is_none());
        clear_env();
    }

    #[test]
    #[serial]
    fn numrs2_prefix_wins_over_alias() {
        clear_env();
        std::env::set_var("NUMRS2_WORLD_SIZE", "4");
        std::env::set_var("NUMRS_WORLD_SIZE", "99");
        let cfg = BootstrapConfig::from_env().expect("valid");
        assert_eq!(cfg.world_size, 4);
        clear_env();
    }

    #[test]
    #[serial]
    fn alias_used_when_canonical_unset() {
        clear_env();
        std::env::set_var("NUMRS_RANK", "2");
        std::env::set_var("NUMRS_WORLD_SIZE", "4");
        let cfg = BootstrapConfig::from_env().expect("valid");
        assert_eq!(cfg.rank, 2);
        assert_eq!(cfg.world_size, 4);
        assert!(!cfg.is_single_process());
        clear_env();
    }

    #[test]
    #[serial]
    fn rejects_rank_out_of_range() {
        clear_env();
        std::env::set_var("NUMRS2_RANK", "5");
        std::env::set_var("NUMRS2_WORLD_SIZE", "4");
        let err = BootstrapConfig::from_env();
        assert!(matches!(
            err,
            Err(NetError::InvalidRank { rank: 5, size: 4 })
        ));
        clear_env();
    }

    #[test]
    #[serial]
    fn rejects_zero_world_size() {
        clear_env();
        std::env::set_var("NUMRS2_WORLD_SIZE", "0");
        let err = BootstrapConfig::from_env();
        assert!(matches!(err, Err(NetError::Bootstrap(_))));
        clear_env();
    }

    #[test]
    #[serial]
    fn parses_addr_list_in_rank_order() {
        clear_env();
        std::env::set_var("NUMRS2_WORLD_SIZE", "2");
        std::env::set_var("NUMRS2_ADDRS", "127.0.0.1:5000, 127.0.0.1:5001");
        let cfg = BootstrapConfig::from_env().expect("valid");
        assert_eq!(cfg.addrs.len(), 2);
        assert_eq!(cfg.addrs[0].port(), 5000);
        assert_eq!(cfg.addrs[1].port(), 5001);
        clear_env();
    }

    #[test]
    #[serial]
    fn rejects_addr_list_length_mismatch() {
        clear_env();
        std::env::set_var("NUMRS2_WORLD_SIZE", "3");
        std::env::set_var("NUMRS2_ADDRS", "127.0.0.1:5000, 127.0.0.1:5001");
        let err = BootstrapConfig::from_env();
        assert!(matches!(err, Err(NetError::Bootstrap(_))));
        clear_env();
    }

    #[test]
    #[serial]
    fn parses_master_addr() {
        clear_env();
        std::env::set_var("NUMRS2_MASTER_ADDR", "10.0.0.1:9000");
        let cfg = BootstrapConfig::from_env().expect("valid");
        assert_eq!(
            cfg.master_addr,
            Some("10.0.0.1:9000".parse().expect("valid literal"))
        );
        clear_env();
    }

    #[test]
    #[serial]
    fn rejects_unparseable_rank() {
        clear_env();
        std::env::set_var("NUMRS2_RANK", "not-a-number");
        let err = BootstrapConfig::from_env();
        assert!(matches!(err, Err(NetError::Bootstrap(_))));
        clear_env();
    }

    // ===============================================================
    // Rendezvous, end to end.
    //
    // `InProcess` is exercised by every `super::super::testing::LocalCluster`
    // test; the two modes a real multi-process launch uses get their own
    // coverage here.
    // ===============================================================

    /// `count` loopback addresses the OS has just confirmed are free
    /// *simultaneously*.
    ///
    /// `Master` and `Static` rendezvous both need addresses agreed before
    /// anyone binds them, so port `0` cannot be used directly. Holding every
    /// listener at once before releasing them is what makes the addresses
    /// distinct — binding and dropping one at a time can hand back the same
    /// port twice. The gap between release and the real bind is a theoretical
    /// race with other processes on the machine; on loopback ephemeral ports
    /// it is small enough to live with in a test.
    async fn reserve_addrs(count: usize) -> Vec<SocketAddr> {
        let mut listeners = Vec::with_capacity(count);
        for _ in 0..count {
            listeners.push(
                TcpListener::bind(("127.0.0.1", 0))
                    .await
                    .expect("bind an ephemeral port"),
            );
        }
        let addrs = listeners
            .iter()
            .map(|l| l.local_addr().expect("local_addr"))
            .collect();
        drop(listeners);
        addrs
    }

    fn ephemeral() -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], 0))
    }

    /// Every rank sends one message around a ring and receives from its
    /// predecessor: proof that the table the rendezvous produced actually
    /// meshed, not just that it parsed.
    async fn ring_exchange(
        rank: u32,
        world_size: u32,
        endpoint: &Endpoint,
    ) -> Result<Vec<u8>, NetError> {
        let next = (rank + 1) % world_size;
        let prev = (rank + world_size - 1) % world_size;
        let payload = vec![rank as u8; 16];
        endpoint
            .send_bytes(next, 0, 1, &payload, SendOpts::default())
            .await?;
        endpoint.recv_bytes(prev, 0, 1).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn single_process_world_short_circuits_to_loopback_only() {
        let endpoint = bootstrap(
            0,
            1,
            RendezvousMode::Static { addrs: Vec::new() },
            ephemeral(),
            EndpointConfig::default(),
        )
        .await
        .expect("a world of one needs no rendezvous at all");

        assert_eq!(endpoint.link_count().await, 0, "nobody to dial");
        endpoint
            .send_bytes(0, 1, 2, b"self", SendOpts::default())
            .await
            .expect("self send");
        assert_eq!(
            endpoint.recv_bytes(0, 1, 2).await.expect("self recv"),
            b"self".to_vec()
        );
        endpoint.shutdown().await.expect("shutdown");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn master_rendezvous_meshes_every_rank() {
        const WORLD: u32 = 3;
        let master = reserve_addrs(1).await.remove(0);

        let mut tasks = Vec::with_capacity(WORLD as usize);
        for rank in 0..WORLD {
            tasks.push(tokio::spawn(async move {
                let endpoint = bootstrap(
                    rank,
                    WORLD,
                    RendezvousMode::Master { addr: master },
                    ephemeral(),
                    EndpointConfig::default(),
                )
                .await?;
                let got = ring_exchange(rank, WORLD, &endpoint).await?;
                endpoint.shutdown().await?;
                Ok::<Vec<u8>, NetError>(got)
            }));
        }

        for (rank, task) in tasks.into_iter().enumerate() {
            let got = tokio::time::timeout(Duration::from_secs(30), task)
                .await
                .unwrap_or_else(|_| panic!("rank {rank} never finished the master rendezvous"))
                .expect("join")
                .unwrap_or_else(|e| panic!("rank {rank} failed the master rendezvous: {e}"));
            let prev = (rank as u32 + WORLD - 1) % WORLD;
            assert_eq!(got, vec![prev as u8; 16]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn static_rendezvous_binds_the_table_it_was_given() {
        const WORLD: u32 = 2;
        let addrs = reserve_addrs(WORLD as usize).await;

        let mut tasks = Vec::with_capacity(WORLD as usize);
        for rank in 0..WORLD {
            let table = addrs.clone();
            tasks.push(tokio::spawn(async move {
                let expected = table[rank as usize];
                let endpoint = bootstrap(
                    rank,
                    WORLD,
                    RendezvousMode::Static {
                        addrs: table.clone(),
                    },
                    // Deliberately not the table's address: Static mode must
                    // bind what the table says, ignoring this argument.
                    ephemeral(),
                    EndpointConfig::default(),
                )
                .await?;
                if endpoint.local_addr() != expected {
                    return Err(NetError::Bootstrap(format!(
                        "rank {rank} bound {} but the table says {expected}",
                        endpoint.local_addr()
                    )));
                }
                let got = ring_exchange(rank, WORLD, &endpoint).await?;
                endpoint.shutdown().await?;
                Ok::<Vec<u8>, NetError>(got)
            }));
        }

        for (rank, task) in tasks.into_iter().enumerate() {
            let got = tokio::time::timeout(Duration::from_secs(30), task)
                .await
                .unwrap_or_else(|_| panic!("rank {rank} never finished the static rendezvous"))
                .expect("join")
                .unwrap_or_else(|e| panic!("rank {rank} failed the static rendezvous: {e}"));
            let prev = (rank as u32 + WORLD - 1) % WORLD;
            assert_eq!(got, vec![prev as u8; 16]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bootstrap_rejects_a_rank_outside_the_world() {
        let err = bootstrap(
            3,
            2,
            RendezvousMode::Static { addrs: Vec::new() },
            ephemeral(),
            EndpointConfig::default(),
        )
        .await
        .expect_err("rank 3 is not part of a world of 2");
        assert!(matches!(err, NetError::InvalidRank { rank: 3, size: 2 }));
    }

    #[test]
    fn address_table_round_trips_through_the_wire_encoding() {
        let addrs: Vec<SocketAddr> = vec![
            "127.0.0.1:5000".parse().expect("literal"),
            "127.0.0.1:5001".parse().expect("literal"),
        ];
        let encoded = encode_table(addrs.iter().copied());
        assert_eq!(encoded, "127.0.0.1:5000,127.0.0.1:5001");
        assert_eq!(decode_table(encoded.as_bytes()).expect("decodes"), addrs);
    }

    #[test]
    fn rendezvous_mode_prefers_a_static_table_over_a_master() {
        let cfg = BootstrapConfig {
            rank: 0,
            world_size: 2,
            addrs: vec![
                "127.0.0.1:5000".parse().expect("literal"),
                "127.0.0.1:5001".parse().expect("literal"),
            ],
            master_addr: Some("127.0.0.1:6000".parse().expect("literal")),
        };
        assert!(matches!(
            cfg.rendezvous_mode(),
            Ok(RendezvousMode::Static { .. })
        ));
    }

    #[test]
    fn multi_rank_world_without_any_addressing_is_rejected() {
        let cfg = BootstrapConfig {
            rank: 0,
            world_size: 4,
            addrs: Vec::new(),
            master_addr: None,
        };
        assert!(matches!(cfg.rendezvous_mode(), Err(NetError::Bootstrap(_))));
    }
}
