//! In-process multi-rank test harness for the `distributed` module.
//!
//! [`LocalCluster::run`] drives `world_size` copies of one async closure as
//! tokio tasks inside a single process. Each copy gets a [`RankContext`]
//! carrying its rank, the world size, an already-bound ephemeral
//! `127.0.0.1` [`TcpListener`], and a [`Rendezvous`] handle for discovering
//! every other rank's bound address.
//!
//! [`LocalCluster::run_connected`] goes one step further and hands each rank
//! a fully meshed [`Endpoint`]: bind, `InProcess` rendezvous, and
//! [`Endpoint::connect_mesh`] have all completed before the body runs, so a
//! test can send on the first line without a handshake of its own.
//!
//! The rendezvous is in-memory but the *transport is not*: ranks connect to
//! each other over real loopback TCP, through the same framing, queueing and
//! connection code a multi-host run uses. A test that passes here is
//! therefore evidence about the real transport, not about a shortcut.
//!
//! A run that doesn't finish within its deadline ([`GLOBAL_TIMEOUT`] by
//! default) fails with [`ClusterError::Timeout`] naming exactly the ranks
//! that never completed, rather than a bare timeout with no way to tell
//! which rank hung.

use super::bootstrap::InProcessRegistry;
use super::net::endpoint::Endpoint;
use super::net::{EndpointConfig, NetError};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::task::Id;

/// Default deadline for [`LocalCluster::run`]. Tests that specifically
/// exercise the timeout path should call
/// [`LocalCluster::run_with_timeout`] with a much shorter one instead of
/// waiting out the real default.
pub const GLOBAL_TIMEOUT: Duration = Duration::from_secs(20);

/// Shared registry letting every rank in a [`LocalCluster::run`] discover
/// every other rank's bound address.
///
/// This is [`super::bootstrap::InProcessRegistry`] — the production
/// `InProcess` rendezvous back-end — not a test-only stand-in, so the
/// harness bootstraps through exactly the path
/// [`super::bootstrap::bootstrap`] takes in
/// [`super::bootstrap::RendezvousMode::InProcess`] mode.
pub type Rendezvous = InProcessRegistry;

/// Errors from driving a [`LocalCluster`] run.
#[derive(Error, Debug)]
pub enum ClusterError {
    /// Setting up the cluster itself failed (e.g. binding a listener).
    #[error("cluster setup failed: {0}")]
    Setup(NetError),

    /// One rank's body returned an error.
    #[error("rank {rank} failed: {source}")]
    RankFailed { rank: u32, source: NetError },

    /// A rank's task panicked or was cancelled before returning.
    #[error("rank {rank} did not complete cleanly: {message}")]
    RankPanicked { rank: u32, message: String },

    /// The run did not finish within its deadline.
    #[error("timed out after {timeout:?} waiting for rank(s) {stuck:?} to complete")]
    Timeout { timeout: Duration, stuck: Vec<u32> },
}

/// What [`LocalCluster::run`] hands to each rank's body.
pub struct RankContext {
    /// This rank, `0..world_size`.
    pub rank: u32,
    /// Total number of ranks in this run.
    pub world_size: u32,
    /// Registry for discovering every rank's bound address.
    pub rendezvous: Rendezvous,
    /// This rank's own listener, already bound to `127.0.0.1` on an
    /// OS-assigned ephemeral port, and already published to
    /// [`Self::rendezvous`].
    ///
    /// Build an [`Endpoint`] with [`Self::into_endpoint`] rather than
    /// binding a fresh socket: peers have already been told *this* port, so a
    /// second socket would leave them dialing an address nobody accepts on.
    pub listener: TcpListener,
}

impl RankContext {
    /// Turn this rank's pre-bound listener into a fully meshed [`Endpoint`].
    ///
    /// Adopts [`Self::listener`] (so the address peers were given stays the
    /// address this rank accepts on), waits for every rank to publish through
    /// the in-process rendezvous, and completes
    /// [`Endpoint::connect_mesh`] before returning.
    pub async fn into_endpoint(self, config: EndpointConfig) -> Result<Endpoint, NetError> {
        let endpoint = Endpoint::from_listener(self.listener, self.rank, self.world_size, config)?;
        let addrs = self.rendezvous.all_addrs().await;
        endpoint.connect_mesh(&addrs).await?;
        Ok(endpoint)
    }
}

/// What [`LocalCluster::run_connected`] hands to each rank's body: the same
/// identity as [`RankContext`], with the listener already turned into a
/// connected [`Endpoint`].
pub struct ClusterNode {
    /// This rank, `0..world_size`.
    pub rank: u32,
    /// Total number of ranks in this run.
    pub world_size: u32,
    /// This rank's endpoint, meshed to every peer.
    pub endpoint: Endpoint,
}

impl ClusterNode {
    /// The rank one step around the ring from this one.
    pub fn next_rank(&self) -> u32 {
        (self.rank + 1) % self.world_size
    }

    /// The rank one step back around the ring from this one.
    pub fn prev_rank(&self) -> u32 {
        (self.rank + self.world_size - 1) % self.world_size
    }
}

/// In-process multi-rank test cluster. See the module docs.
pub struct LocalCluster;

impl LocalCluster {
    /// Run `body` once per rank in `0..world_size`, as concurrent tokio
    /// tasks in the current runtime, waiting up to [`GLOBAL_TIMEOUT`] for
    /// all of them to finish. Results are returned in rank order.
    pub async fn run<F, Fut, T>(world_size: u32, body: F) -> Result<Vec<T>, ClusterError>
    where
        F: Fn(RankContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T, NetError>> + Send + 'static,
        T: Send + 'static,
    {
        Self::run_with_timeout(world_size, GLOBAL_TIMEOUT, body).await
    }

    /// As [`Self::run`], with an explicit deadline instead of
    /// [`GLOBAL_TIMEOUT`].
    pub async fn run_with_timeout<F, Fut, T>(
        world_size: u32,
        timeout: Duration,
        body: F,
    ) -> Result<Vec<T>, ClusterError>
    where
        F: Fn(RankContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T, NetError>> + Send + 'static,
        T: Send + 'static,
    {
        if world_size == 0 {
            return Err(ClusterError::Setup(NetError::Bootstrap(
                "world_size must be >= 1".to_string(),
            )));
        }

        let rendezvous = Rendezvous::new(world_size);
        let mut listeners = Vec::with_capacity(world_size as usize);
        for rank in 0..world_size {
            let listener = TcpListener::bind(("127.0.0.1", 0))
                .await
                .map_err(|e| ClusterError::Setup(NetError::Io(e.to_string())))?;
            let addr = listener
                .local_addr()
                .map_err(|e| ClusterError::Setup(NetError::Io(e.to_string())))?;
            rendezvous.register(rank, addr);
            listeners.push(listener);
        }

        let body = Arc::new(body);
        let mut set = tokio::task::JoinSet::new();
        let mut id_to_rank: HashMap<Id, u32> = HashMap::with_capacity(world_size as usize);
        for (rank, listener) in (0..world_size).zip(listeners) {
            let ctx = RankContext {
                rank,
                world_size,
                rendezvous: rendezvous.clone(),
                listener,
            };
            let body_for_rank = Arc::clone(&body);
            let handle = set.spawn(body_for_rank(ctx));
            id_to_rank.insert(handle.id(), rank);
        }

        let mut remaining: BTreeSet<u32> = (0..world_size).collect();
        let mut results: BTreeMap<u32, T> = BTreeMap::new();
        let sleep = tokio::time::sleep(timeout);
        tokio::pin!(sleep);

        loop {
            if remaining.is_empty() {
                break;
            }
            tokio::select! {
                joined = set.join_next_with_id() => {
                    match joined {
                        Some(Ok((id, Ok(value)))) => {
                            let rank = id_to_rank.get(&id).copied().unwrap_or(u32::MAX);
                            remaining.remove(&rank);
                            results.insert(rank, value);
                        }
                        Some(Ok((id, Err(source)))) => {
                            let rank = id_to_rank.get(&id).copied().unwrap_or(u32::MAX);
                            set.abort_all();
                            return Err(ClusterError::RankFailed { rank, source });
                        }
                        Some(Err(join_err)) => {
                            let rank = id_to_rank.get(&join_err.id()).copied().unwrap_or(u32::MAX);
                            set.abort_all();
                            return Err(ClusterError::RankPanicked {
                                rank,
                                message: join_err.to_string(),
                            });
                        }
                        None => break,
                    }
                }
                _ = &mut sleep => {
                    set.abort_all();
                    return Err(ClusterError::Timeout {
                        timeout,
                        stuck: remaining.into_iter().collect(),
                    });
                }
            }
        }

        Ok(results.into_values().collect())
    }

    /// Run `body` once per rank with a fully connected [`Endpoint`] already
    /// in hand. Uses [`EndpointConfig::default`] and [`GLOBAL_TIMEOUT`].
    pub async fn run_connected<F, Fut, T>(world_size: u32, body: F) -> Result<Vec<T>, ClusterError>
    where
        F: Fn(ClusterNode) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T, NetError>> + Send + 'static,
        T: Send + 'static,
    {
        Self::run_connected_with(world_size, EndpointConfig::default(), GLOBAL_TIMEOUT, body).await
    }

    /// As [`Self::run_connected`], with an explicit endpoint configuration
    /// and deadline.
    ///
    /// Every rank's endpoint is held alive by the harness until the whole run
    /// finishes, so a rank that returns early can never tear down a link a
    /// slower peer is still using; they are all shut down together at the end.
    pub async fn run_connected_with<F, Fut, T>(
        world_size: u32,
        config: EndpointConfig,
        timeout: Duration,
        body: F,
    ) -> Result<Vec<T>, ClusterError>
    where
        F: Fn(ClusterNode) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<T, NetError>> + Send + 'static,
        T: Send + 'static,
    {
        let body = Arc::new(body);
        let alive: Arc<Mutex<Vec<Endpoint>>> = Arc::new(Mutex::new(Vec::new()));
        let alive_for_run = Arc::clone(&alive);

        let outcome = Self::run_with_timeout(world_size, timeout, move |ctx: RankContext| {
            let body = Arc::clone(&body);
            let alive = Arc::clone(&alive_for_run);
            let config = config.clone();
            async move {
                let rank = ctx.rank;
                let world_size = ctx.world_size;
                let endpoint = ctx.into_endpoint(config).await?;
                match alive.lock() {
                    Ok(mut guard) => guard.push(endpoint.clone()),
                    Err(poisoned) => poisoned.into_inner().push(endpoint.clone()),
                }
                body(ClusterNode {
                    rank,
                    world_size,
                    endpoint,
                })
                .await
            }
        })
        .await;

        let endpoints: Vec<Endpoint> = match alive.lock() {
            Ok(mut guard) => guard.drain(..).collect(),
            Err(poisoned) => poisoned.into_inner().drain(..).collect(),
        };
        for endpoint in &endpoints {
            let _ = endpoint.shutdown().await;
        }
        outcome
    }
}

/// Deterministic pseudo-random bytes, for payloads LZ4 cannot shrink.
///
/// Deliberately not `scirs2_core::random`: this needs a fixed, dependency-free
/// bit pattern that is identical on every run and every platform, so a
/// compression-path assertion can never turn flaky.
pub fn incompressible_bytes(len: usize) -> Vec<u8> {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::net::SendOpts;

    /// A config whose per-operation deadlines expire comfortably *before* the
    /// harness deadline the deadlock tests run under.
    ///
    /// The ordering matters for diagnosis, not for pass/fail: whichever clock
    /// fires first decides what the failure looks like. With the endpoint's
    /// recv timeout first, a genuine stall is reported as
    /// [`NetError::RecvTimeout`] naming the exact `(src, ctx, tag)` that never
    /// arrived; with the harness first, all anyone learns is "rank N did not
    /// finish". These are generous enough that a merely-slow machine (many
    /// agents compiling at once) still passes.
    fn impatient_config() -> EndpointConfig {
        EndpointConfig {
            send_timeout: Duration::from_secs(20),
            recv_timeout: Duration::from_secs(20),
            ..EndpointConfig::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn local_cluster_runs_all_ranks_in_order() {
        let results = LocalCluster::run(3, |ctx: RankContext| async move { Ok(ctx.rank) })
            .await
            .expect("cluster run should succeed");
        assert_eq!(results, vec![0, 1, 2]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rendezvous_lets_ranks_discover_each_other() {
        let results = LocalCluster::run(2, |ctx: RankContext| async move {
            let peer_rank = if ctx.rank == 0 { 1 } else { 0 };
            let peer_addr = ctx.rendezvous.lookup(peer_rank).await;
            let own_addr = ctx
                .listener
                .local_addr()
                .map_err(|e| NetError::Io(e.to_string()))?;
            Ok((own_addr, peer_addr))
        })
        .await
        .expect("cluster run should succeed");

        assert_eq!(results.len(), 2);
        // Rank 0 sees rank 1's address as its peer, and vice versa.
        assert_eq!(results[0].1, results[1].0);
        assert_eq!(results[1].1, results[0].0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn all_addrs_waits_for_every_rank() {
        let results = LocalCluster::run(4, |ctx: RankContext| async move {
            let all = ctx.rendezvous.all_addrs().await;
            Ok(all.len())
        })
        .await
        .expect("cluster run should succeed");
        assert_eq!(results, vec![4, 4, 4, 4]);
    }

    #[tokio::test]
    async fn rank_error_is_reported_with_its_rank() {
        let err = LocalCluster::run(2, |ctx: RankContext| async move {
            if ctx.rank == 1 {
                Err(NetError::NotImplemented("boom".to_string()))
            } else {
                Ok(())
            }
        })
        .await
        .expect_err("one rank failing should fail the run");
        assert!(matches!(err, ClusterError::RankFailed { rank: 1, .. }));
    }

    #[tokio::test]
    async fn timeout_names_exactly_the_stuck_rank() {
        // The deadline only needs to comfortably outlast rank 0's
        // effectively-instant completion; rank 1's sleep only needs to
        // comfortably outlast the deadline (it is aborted the moment the
        // deadline fires, so making it longer does not slow the test down).
        let err = LocalCluster::run_with_timeout(
            2,
            Duration::from_secs(1),
            |ctx: RankContext| async move {
                if ctx.rank == 1 {
                    tokio::time::sleep(Duration::from_secs(60)).await;
                }
                Ok(())
            },
        )
        .await
        .expect_err("should time out");

        match err {
            ClusterError::Timeout { stuck, .. } => assert_eq!(stuck, vec![1]),
            other => panic!("expected ClusterError::Timeout, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_zero_world_size() {
        let err = LocalCluster::run(0, |ctx: RankContext| async move { Ok(ctx.rank) })
            .await
            .expect_err("world_size 0 should be rejected");
        assert!(matches!(err, ClusterError::Setup(_)));
    }

    // ===============================================================
    // Transport tests over a real (loopback TCP) mesh.
    // ===============================================================

    /// Every rank sends around a ring and receives from its predecessor,
    /// for worlds of 2, 3 and 4.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn point_to_point_round_trip_for_worlds_of_two_to_four() {
        for world_size in 2..=4u32 {
            let results = LocalCluster::run_connected(world_size, |node: ClusterNode| async move {
                let payload = vec![node.rank as u8; 32];
                node.endpoint
                    .send_bytes(node.next_rank(), 0, 1, &payload, SendOpts::default())
                    .await?;
                let got = node.endpoint.recv_bytes(node.prev_rank(), 0, 1).await?;
                Ok(got)
            })
            .await
            .unwrap_or_else(|e| panic!("world of {world_size} should run: {e}"));

            assert_eq!(results.len(), world_size as usize);
            for (rank, got) in results.iter().enumerate() {
                let expected_src = (rank as u32 + world_size - 1) % world_size;
                assert_eq!(
                    got,
                    &vec![expected_src as u8; 32],
                    "rank {rank} in a world of {world_size} got the wrong sender's payload"
                );
            }
        }
    }

    /// Tag 2 is sent first, but the receiver asks for tag 1 first: the
    /// mailbox must demultiplex rather than hand over whatever arrived.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tagged_delivery_is_demultiplexed_out_of_order() {
        let results = LocalCluster::run_connected(2, |node: ClusterNode| async move {
            if node.rank == 0 {
                node.endpoint
                    .send_bytes(1, 0, 2, b"tag-two", SendOpts::default())
                    .await?;
                node.endpoint
                    .send_bytes(1, 0, 1, b"tag-one", SendOpts::default())
                    .await?;
                Ok(Vec::new())
            } else {
                // Deliberately the reverse of the send order.
                let one = node.endpoint.recv_bytes(0, 0, 1).await?;
                let two = node.endpoint.recv_bytes(0, 0, 2).await?;
                Ok(vec![one, two])
            }
        })
        .await
        .expect("run");

        assert_eq!(results[1], vec![b"tag-one".to_vec(), b"tag-two".to_vec()]);
    }

    /// A payload well over the compression threshold must survive the LZ4
    /// round trip byte-for-byte, both when it compresses well and when it is
    /// incompressible (LZ4 would *expand* it, so it goes raw instead).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn large_frames_round_trip_compressible_and_incompressible() {
        const BIG: usize = 4 * 1024 * 1024;
        let results = LocalCluster::run_connected(2, |node: ClusterNode| async move {
            let compressible = vec![0xABu8; BIG];
            let random = incompressible_bytes(BIG);
            if node.rank == 0 {
                node.endpoint
                    .send_bytes(1, 0, 10, &compressible, SendOpts { compress: true })
                    .await?;
                node.endpoint
                    .send_bytes(1, 0, 11, &random, SendOpts { compress: true })
                    .await?;
                Ok(true)
            } else {
                let got_compressible = node.endpoint.recv_bytes(0, 0, 10).await?;
                let got_random = node.endpoint.recv_bytes(0, 0, 11).await?;
                Ok(got_compressible == compressible && got_random == random)
            }
        })
        .await
        .expect("run");
        assert_eq!(results, vec![true, true]);
    }

    /// `SendOpts::default()` (compress: false) must stay uncompressed even
    /// for a payload far above the threshold — a latency probe has to be able
    /// to opt out.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compression_can_be_opted_out_of_for_a_large_payload() {
        const BIG: usize = 512 * 1024;
        let results = LocalCluster::run_connected(2, |node: ClusterNode| async move {
            let payload = vec![0x5Au8; BIG];
            if node.rank == 0 {
                node.endpoint
                    .send_bytes(1, 0, 1, &payload, SendOpts { compress: false })
                    .await?;
                Ok(true)
            } else {
                let got = node.endpoint.recv_bytes(0, 0, 1).await?;
                Ok(got == payload)
            }
        })
        .await
        .expect("run");
        assert_eq!(results, vec![true, true]);
    }

    /// A rank sending to itself never touches a socket.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn self_send_is_a_local_loopback() {
        let results = LocalCluster::run_connected(3, |node: ClusterNode| async move {
            let payload = vec![node.rank as u8; 8];
            node.endpoint
                .send_bytes(node.rank, 4, 5, &payload, SendOpts::default())
                .await?;
            let got = node.endpoint.recv_bytes(node.rank, 4, 5).await?;
            Ok(got == payload)
        })
        .await
        .expect("run");
        assert_eq!(results, vec![true, true, true]);
    }

    /// The regression this whole transport exists for.
    ///
    /// Both ranks push 4 MiB at each other *before* either one starts
    /// receiving. The old `comm::CommunicationChannel` held one
    /// `Arc<Mutex<TcpStream>>` across every `.await`, so this exact
    /// interleaving deadlocked: neither side could reach the write half while
    /// the other was parked in `read_exact`. The payload has to be large
    /// enough to overflow the kernel socket buffers — a small message
    /// completes even under the broken design and proves nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn simultaneous_bidirectional_send_does_not_deadlock() {
        const BIG: usize = 4 * 1024 * 1024;
        // Give the harness clear headroom over the endpoint's own recv
        // deadline. If this ever regresses into a real hang we want the
        // failure to be `RecvTimeout` naming the (src, ctx, tag) that never
        // arrived, not an ambiguous harness-level `ClusterError::Timeout`.
        let results = LocalCluster::run_connected_with(
            2,
            impatient_config(),
            Duration::from_secs(60),
            |node: ClusterNode| async move {
                let peer = node.next_rank();
                let outgoing = vec![node.rank as u8; BIG];

                // Both ranks send first, and only then receive. Send must
                // therefore be enqueue-and-return, not write-and-block.
                node.endpoint
                    .send_bytes(peer, 0, 1, &outgoing, SendOpts::default())
                    .await?;
                let incoming = node.endpoint.recv_bytes(peer, 0, 1).await?;

                Ok(incoming == vec![peer as u8; BIG])
            },
        )
        .await
        .expect("bidirectional traffic must complete, not deadlock");
        assert_eq!(results, vec![true, true]);
    }

    /// The same shape with every rank of a 4-way world blasting every other
    /// rank before receiving anything.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn all_to_all_send_before_recv_completes() {
        const CHUNK: usize = 512 * 1024;
        let results = LocalCluster::run_connected_with(
            4,
            impatient_config(),
            Duration::from_secs(60),
            |node: ClusterNode| async move {
                for peer in 0..node.world_size {
                    if peer == node.rank {
                        continue;
                    }
                    let payload = vec![node.rank as u8; CHUNK];
                    node.endpoint
                        .send_bytes(peer, 0, 7, &payload, SendOpts::default())
                        .await?;
                }
                let mut received = 0usize;
                for peer in 0..node.world_size {
                    if peer == node.rank {
                        continue;
                    }
                    let got = node.endpoint.recv_bytes(peer, 0, 7).await?;
                    if got == vec![peer as u8; CHUNK] {
                        received += 1;
                    }
                }
                Ok(received)
            },
        )
        .await
        .expect("all-to-all must complete");
        assert_eq!(results, vec![3, 3, 3, 3]);
    }

    /// A receive that nothing satisfies must fail with the key named, not
    /// hang and not fail anonymously.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recv_timeout_names_the_missing_src_ctx_tag() {
        let results = LocalCluster::run_connected(2, |node: ClusterNode| async move {
            if node.rank == 0 {
                return Ok(String::new());
            }
            let err = node
                .endpoint
                .recv_bytes_timeout(0, 314, 271, Duration::from_millis(100))
                .await
                .err()
                .ok_or_else(|| {
                    NetError::MalformedFrame("expected the recv to time out".to_string())
                })?;
            match err {
                NetError::RecvTimeout { src, ctx, tag, .. } => {
                    if (src, ctx, tag) != (0, 314, 271) {
                        return Err(NetError::MalformedFrame(format!(
                            "timeout named ({src}, {ctx}, {tag})"
                        )));
                    }
                    Ok(err.to_string())
                }
                other => Err(other),
            }
        })
        .await
        .expect("run");

        let message = &results[1];
        assert!(message.contains("src 0"), "{message}");
        assert!(message.contains("ctx 314"), "{message}");
        assert!(message.contains("tag 271"), "{message}");
    }

    /// FIFO ordering within one key, across many messages.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn messages_under_one_key_arrive_in_order() {
        const COUNT: usize = 64;
        let results = LocalCluster::run_connected(2, |node: ClusterNode| async move {
            if node.rank == 0 {
                for i in 0..COUNT {
                    node.endpoint
                        .send_owned(1, 0, 1, vec![i as u8], SendOpts::default())
                        .await?;
                }
                Ok(Vec::new())
            } else {
                let mut got = Vec::with_capacity(COUNT);
                for _ in 0..COUNT {
                    let payload = node.endpoint.recv_bytes(0, 0, 1).await?;
                    got.push(payload.first().copied().unwrap_or(u8::MAX));
                }
                Ok(got)
            }
        })
        .await
        .expect("run");

        let expected: Vec<u8> = (0..COUNT).map(|i| i as u8).collect();
        assert_eq!(results[1], expected);
    }

    #[test]
    fn incompressible_bytes_are_actually_incompressible() {
        let data = incompressible_bytes(256 * 1024);
        assert_eq!(data.len(), 256 * 1024);
        let compressed = oxiarc_lz4::block::compress_block(&data).expect("compress");
        assert!(
            compressed.len() >= data.len(),
            "test fixture must defeat LZ4: {} vs {}",
            compressed.len(),
            data.len()
        );
    }
}
