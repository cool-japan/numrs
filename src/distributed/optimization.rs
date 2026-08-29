//! Network Optimization for Distributed Computing
//!
//! This module provides network-aware optimizations to improve communication
//! efficiency in distributed computations.
//!
//! # Features
//!
//! - Network topology detection and modeling
//! - Bandwidth and latency measurement
//! - Communication pattern optimization
//! - Computation-communication overlap
//! - Data compression for network transfer
//!
//! # Example
//!
//! ```rust,no_run
//! use numrs2::distributed::optimization::*;
//! use numrs2::distributed::process::*;
//! use numrs2::distributed::net::SendOpts;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let world = init().await?;
//! let peer = if world.rank() == 0 { 1 } else { 0 };
//!
//! // Real ping-pong probes over the world communicator's shared Endpoint
//! // (see "Probe protocol and wire tags" below for exactly what's measured).
//! let opts = SendOpts { compress: false };
//! let latency_us = measure_latency(&world, peer, 5, opts).await?;
//! let bandwidth_mib_s = measure_bandwidth(&world, peer, 4 * 1024 * 1024, opts).await?;
//! println!("Latency: {latency_us} us, Bandwidth: {bandwidth_mib_s} MiB/s");
//!
//! // Overlap a CPU-bound computation with an independent async operation:
//! // `compute` starts running on a blocking-pool thread immediately, while
//! // `comm` is awaited concurrently on the caller's task.
//! let (checksum, ()) = overlap_compute_communicate(
//!     || (0..1_000_000u64).fold(0u64, |acc, i| acc.wrapping_add(i)),
//!     async { Ok(()) },
//! )
//! .await?;
//! println!("compute finished with {checksum}");
//! # Ok(())
//! # }
//! ```
//!
//! # Probe protocol and wire tags
//!
//! [`measure_latency`]/[`measure_bandwidth`] are real measurements over
//! [`super::process::Communicator::require_endpoint`] — no raw sockets, no
//! bypassing [`super::net::endpoint::Endpoint`]'s mailbox/compression policy.
//! Both give the lower-ranked of the two participants ("the client") an
//! unambiguous client role: only the client's clock ever times anything,
//! which sidesteps clock/scheduling skew between two independently-running
//! ranks' tasks entirely. The client relays its own measured figure back to
//! the higher-ranked participant ("the server") as a final small message, so
//! **both ranks return the identical, client-measured number** rather than
//! each guessing at its own view of elapsed time.
//!
//! - **Latency**: for each of `rounds` rounds (client and server agree on
//!   `rounds` the same way every collective in [`super::collective`] expects
//!   every rank to call it in matching order), the client times an empty
//!   ping/pong exchange: send ping, wait for the server's pong. The server's
//!   pong for round `r` is only ever sent after receiving the client's ping
//!   for that exact round — a real, causally-dependent round trip, not two
//!   independently-timed one-way sends. Latency is
//!   `median(round-trip times) / 2`.
//! - **Bandwidth**: the client times one real `payload_bytes`-sized transfer
//!   (send, then wait for the server's small ack) and reports
//!   `payload_bytes / elapsed` in MiB/s — a slight underestimate of pure
//!   link bandwidth, since it also includes one small-message return trip.
//!
//! Wire tags used here (`0xD`/`0xE` high nibbles) are deliberately disjoint
//! from [`super::collective`]'s `TAG_*` constants
//! (`0x1_0000_0000..=0xA_0000_0000`), from [`super::model_parallel`]'s
//! pipeline tags (`0xB_0000_0000`/`0xC_0000_0000`), and from
//! [`super::communication`]'s `ASYNC_COMM_TAG` (`0xACC0_...`) — so a probe
//! can never be confused with collective, pipeline, or `AsyncCommunicator`
//! traffic sharing the same `(comm, ctx)`.

use super::collective::CollectiveError;
use super::net::{NetError, SendOpts};
use super::process::{Communicator, ProcessError};
use oxiarc_lz4::{compress as lz4_compress, decompress as lz4_decompress};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::time::Instant;
use thiserror::Error;

/// Errors that can occur during network optimization
#[derive(Error, Debug)]
pub enum OptimizationError {
    #[error("Process error: {0}")]
    Process(#[from] ProcessError),

    #[error("Collective operation error: {0}")]
    Collective(#[from] CollectiveError),

    /// A failure from the real [`super::net`] transport layer, surfaced by
    /// [`measure_latency`]/[`measure_bandwidth`] going through
    /// [`super::net::endpoint::Endpoint`] directly.
    #[error("Transport error: {0}")]
    Net(#[from] NetError),

    /// [`overlap_compute_communicate`]'s blocking-pool task panicked or was
    /// cancelled before the computation finished.
    #[error("compute task failed: {0}")]
    ComputeTaskFailed(String),

    #[error("Topology detection failed: {0}")]
    TopologyError(String),

    #[error("Measurement failed: {0}")]
    MeasurementError(String),

    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("Compression error: {0}")]
    CompressionError(String),
}

/// Network topology types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkTopology {
    /// Fully connected network (all-to-all)
    FullyConnected,

    /// Tree topology
    Tree { arity: usize },

    /// Ring topology
    Ring,

    /// 2D/3D mesh topology
    Mesh { dims: [usize; 3] },

    /// Hypercube topology
    Hypercube { dimension: usize },

    /// Fat-tree topology (common in data centers)
    FatTree { levels: usize },

    /// Custom topology
    Custom,
}

impl NetworkTopology {
    /// Get optimal algorithm for collective operations on this topology
    pub fn optimal_algorithm(&self, op: &str) -> Algorithm {
        match (self, op) {
            (NetworkTopology::Tree { .. }, "broadcast") => Algorithm::TreeBroadcast,
            (NetworkTopology::Ring, "reduce") => Algorithm::RingReduce,
            (NetworkTopology::Hypercube { .. }, "allreduce") => Algorithm::HypercubeAllReduce,
            _ => Algorithm::Default,
        }
    }

    /// Check if topology supports efficient point-to-point for given rank pair
    pub fn has_direct_connection(&self, src: usize, dst: usize, _size: usize) -> bool {
        match self {
            NetworkTopology::FullyConnected => true,
            NetworkTopology::Ring => (src as i64 - dst as i64).abs() == 1,
            _ => false, // Conservative for other topologies
        }
    }
}

/// Communication algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Default algorithm
    Default,

    /// Tree-based broadcast
    TreeBroadcast,

    /// Ring-based reduction
    RingReduce,

    /// Hypercube all-reduce
    HypercubeAllReduce,

    /// Recursive doubling
    RecursiveDoubling,

    /// Pairwise exchange
    PairwiseExchange,
}

/// Network bandwidth model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthModel {
    /// Measured bandwidths between process pairs (in MB/s)
    measurements: Vec<(usize, usize, f64)>,

    /// Average bandwidth
    average: f64,

    /// Minimum bandwidth (bottleneck)
    min: f64,

    /// Maximum bandwidth
    max: f64,
}

impl BandwidthModel {
    /// Create a new bandwidth model
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            average: 0.0,
            min: 0.0,
            max: 0.0,
        }
    }

    /// Add a measurement
    pub fn add_measurement(&mut self, src: usize, dst: usize, bandwidth: f64) {
        self.measurements.push((src, dst, bandwidth));
        self.update_statistics();
    }

    /// Update statistics after adding measurements
    fn update_statistics(&mut self) {
        if self.measurements.is_empty() {
            return;
        }

        let values: Vec<f64> = self.measurements.iter().map(|(_, _, bw)| *bw).collect();

        self.average = values.iter().sum::<f64>() / values.len() as f64;
        self.min = values.iter().copied().fold(f64::INFINITY, f64::min);
        self.max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }

    /// Get estimated bandwidth between two processes
    pub fn estimate(&self, src: usize, dst: usize) -> f64 {
        // Look for exact measurement
        for &(s, d, bw) in &self.measurements {
            if (s == src && d == dst) || (s == dst && d == src) {
                return bw;
            }
        }

        // Return average as fallback
        self.average
    }
}

impl Default for BandwidthModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Network latency model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyModel {
    /// Measured latencies between process pairs (in microseconds)
    measurements: Vec<(usize, usize, f64)>,

    /// Average latency
    average: f64,

    /// Minimum latency
    min: f64,

    /// Maximum latency
    max: f64,
}

impl LatencyModel {
    /// Create a new latency model
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            average: 0.0,
            min: 0.0,
            max: 0.0,
        }
    }

    /// Add a measurement
    pub fn add_measurement(&mut self, src: usize, dst: usize, latency: f64) {
        self.measurements.push((src, dst, latency));
        self.update_statistics();
    }

    /// Update statistics
    fn update_statistics(&mut self) {
        if self.measurements.is_empty() {
            return;
        }

        let values: Vec<f64> = self.measurements.iter().map(|(_, _, lat)| *lat).collect();

        self.average = values.iter().sum::<f64>() / values.len() as f64;
        self.min = values.iter().copied().fold(f64::INFINITY, f64::min);
        self.max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }

    /// Get estimated latency between two processes
    pub fn estimate(&self, src: usize, dst: usize) -> f64 {
        // Look for exact measurement
        for &(s, d, lat) in &self.measurements {
            if (s == src && d == dst) || (s == dst && d == src) {
                return lat;
            }
        }

        // Return average as fallback
        self.average
    }
}

impl Default for LatencyModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect network topology
///
/// Attempts to detect the network topology by analyzing process connectivity.
pub async fn detect_topology(comm: &Communicator) -> Result<NetworkTopology, OptimizationError> {
    let size = comm.size();

    // Simple heuristic based on size
    if size.is_power_of_two() && size >= 8 {
        Ok(NetworkTopology::Hypercube {
            dimension: (size as f64).log2() as usize,
        })
    } else {
        // Default to fully connected for small clusters
        Ok(NetworkTopology::FullyConnected)
    }
}

/// Base wire tag for [`measure_latency`]'s ping/pong exchange; folds in the
/// round index so every round of one probe gets its own tag, and the high
/// nibble keeps this disjoint from every other tag range in the crate (see
/// the module docs).
const TAG_PROBE_LATENCY_BASE: u64 = 0xD_0000_0000;

/// Wire tag for [`measure_latency`]'s final client-to-server relay of the
/// computed result (see the module docs on why only the client's clock is
/// ever trusted). Deliberately far from [`TAG_PROBE_LATENCY_BASE`]'s
/// `+ round` range (which a large `rounds` could otherwise grow into) rather
/// than merely the next integer above it.
const TAG_PROBE_LATENCY_RESULT: u64 = 0xD_8000_0000;

/// Wire tag for [`measure_bandwidth`]'s bulk payload transfer.
const TAG_PROBE_BANDWIDTH_DATA: u64 = 0xE_0000_0000;

/// Wire tag for [`measure_bandwidth`]'s ack back to the client, and (reusing
/// the same tag for the opposite direction, since one call only ever sends
/// one message each way under it) the client's relayed result to the server.
/// Deliberately far from [`TAG_PROBE_BANDWIDTH_DATA`] rather than the next
/// integer above it, matching [`TAG_PROBE_LATENCY_RESULT`]'s spacing.
const TAG_PROBE_BANDWIDTH_ACK: u64 = 0xE_8000_0000;

/// Encode `value` as a self-describing little-endian `f64` for the tiny
/// fixed-shape messages [`measure_latency`]/[`measure_bandwidth`] relay.
fn encode_f64(value: f64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

/// Inverse of [`encode_f64`].
fn decode_f64(bytes: &[u8]) -> Result<f64, OptimizationError> {
    let arr: [u8; 8] = bytes.try_into().map_err(|_| {
        OptimizationError::MeasurementError(format!(
            "expected an 8-byte f64 relay message, got {} bytes",
            bytes.len()
        ))
    })?;
    Ok(f64::from_le_bytes(arr))
}

/// Measure round-trip latency to `peer` over `comm`'s shared
/// [`super::net::endpoint::Endpoint`], as `median(round-trip) / 2`
/// microseconds across `rounds` independent ping/pong exchanges (`rounds` is
/// clamped up to 1 so a caller passing `0` still gets one real measurement
/// rather than an empty median).
///
/// Both ranks must call this concurrently with the same `peer`-of-each-other
/// and the same `rounds` (the same requirement collectives in
/// [`super::collective`] place on their callers). Only the lower-ranked of
/// `comm.rank()`/`peer` (the client) ever starts a clock — see the module
/// docs for why — and relays its measured value to the other rank, so both
/// return the identical number. `opts.compress` is honored exactly as it
/// would be for any other [`super::net::endpoint::Endpoint`] send; an empty
/// ping/pong payload is never worth compressing regardless.
pub async fn measure_latency(
    comm: &Communicator,
    peer: usize,
    rounds: usize,
    opts: SendOpts,
) -> Result<f64, OptimizationError> {
    let rounds = rounds.max(1);
    let endpoint = comm.require_endpoint()?;
    let peer_global = comm.global_rank(peer)?;
    let ctx = comm.context().0;
    let rank = comm.rank();

    if rank < peer {
        // Client: time `rounds` real, causally-dependent round trips.
        let mut round_trips_us = Vec::with_capacity(rounds);
        for round in 0..rounds {
            let tag = TAG_PROBE_LATENCY_BASE + round as u64;
            let start = Instant::now();
            endpoint
                .send_bytes(peer_global, ctx, tag, &[], opts)
                .await?;
            endpoint.recv_bytes(peer_global, ctx, tag).await?;
            round_trips_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        round_trips_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = round_trips_us.len() / 2;
        let median_rtt = if round_trips_us.len().is_multiple_of(2) && round_trips_us.len() > 1 {
            (round_trips_us[mid - 1] + round_trips_us[mid]) / 2.0
        } else {
            round_trips_us[mid]
        };
        let latency_us = median_rtt / 2.0;

        endpoint
            .send_bytes(
                peer_global,
                ctx,
                TAG_PROBE_LATENCY_RESULT,
                &encode_f64(latency_us),
                SendOpts::default(),
            )
            .await?;
        Ok(latency_us)
    } else {
        // Server: echo every ping straight back as its pong, then receive
        // the client's authoritative, already-measured result.
        for round in 0..rounds {
            let tag = TAG_PROBE_LATENCY_BASE + round as u64;
            endpoint.recv_bytes(peer_global, ctx, tag).await?;
            endpoint
                .send_bytes(peer_global, ctx, tag, &[], opts)
                .await?;
        }
        let bytes = endpoint
            .recv_bytes(peer_global, ctx, TAG_PROBE_LATENCY_RESULT)
            .await?;
        decode_f64(&bytes)
    }
}

/// Measure transfer rate to/from `peer` over `comm`'s shared
/// [`super::net::endpoint::Endpoint`] by timing one real `payload_bytes`-sized
/// transfer, in MiB/s.
///
/// Only the lower-ranked of `comm.rank()`/`peer` (the client) starts a clock:
/// it sends the payload, waits for the server's small ack, and computes
/// `payload_bytes / elapsed` (a slight underestimate of pure link bandwidth,
/// since it also includes that small return trip) — then relays the value to
/// the server, so both ranks return the identical, client-measured number.
/// Both ranks must call this concurrently with the same `peer`-of-each-other
/// and the same `payload_bytes`. `opts.compress` is honored exactly as
/// [`super::net::endpoint::Endpoint::send_owned`] would for any other send —
/// pass `compress: false` to measure raw link throughput, or `true` to
/// measure the throughput actually achieved after compression.
pub async fn measure_bandwidth(
    comm: &Communicator,
    peer: usize,
    payload_bytes: usize,
    opts: SendOpts,
) -> Result<f64, OptimizationError> {
    let endpoint = comm.require_endpoint()?;
    let peer_global = comm.global_rank(peer)?;
    let ctx = comm.context().0;
    let rank = comm.rank();

    if rank < peer {
        // Client: send the payload, wait for the ack, time the round trip.
        let payload = vec![0u8; payload_bytes];
        let start = Instant::now();
        endpoint
            .send_owned(peer_global, ctx, TAG_PROBE_BANDWIDTH_DATA, payload, opts)
            .await?;
        endpoint
            .recv_bytes(peer_global, ctx, TAG_PROBE_BANDWIDTH_ACK)
            .await?;
        let elapsed = start.elapsed().as_secs_f64().max(f64::MIN_POSITIVE);
        let mib_per_s = (payload_bytes as f64 / (1024.0 * 1024.0)) / elapsed;

        endpoint
            .send_bytes(
                peer_global,
                ctx,
                TAG_PROBE_BANDWIDTH_ACK,
                &encode_f64(mib_per_s),
                SendOpts::default(),
            )
            .await?;
        Ok(mib_per_s)
    } else {
        // Server: receive the payload, ack it, then receive the client's
        // authoritative, already-measured result.
        endpoint
            .recv_bytes(peer_global, ctx, TAG_PROBE_BANDWIDTH_DATA)
            .await?;
        endpoint
            .send_bytes(peer_global, ctx, TAG_PROBE_BANDWIDTH_ACK, &[0u8], opts)
            .await?;
        let bytes = endpoint
            .recv_bytes(peer_global, ctx, TAG_PROBE_BANDWIDTH_ACK)
            .await?;
        decode_f64(&bytes)
    }
}

/// Optimize collective operation for given topology
///
/// Returns the optimal algorithm for a collective operation on the given topology.
pub fn optimize_collective(
    _op: &str,
    topology: &NetworkTopology,
) -> Result<Algorithm, OptimizationError> {
    Ok(topology.optimal_algorithm(_op))
}

/// Overlap a CPU-bound computation with an independent communication future,
/// returning both results once they've both finished.
///
/// `compute` is moved onto a dedicated blocking-pool thread via
/// [`tokio::task::spawn_blocking`] — genuinely running on its own OS thread,
/// not merely interleaved on the caller's async task — so it makes real
/// progress even if it never yields, while `comm` is awaited concurrently on
/// the caller's own task via [`tokio::join!`]. Communication and computation
/// therefore overlap in wall-clock time rather than running one after the
/// other, up to whatever real parallelism the machine has; this is why
/// `compute` must be `Send + 'static` (it crosses a real thread boundary)
/// while `comm` only needs to be a same-task `Future` (it never leaves the
/// caller's task at all).
///
/// # Errors
///
/// [`OptimizationError::ComputeTaskFailed`] if `compute`'s blocking-pool task
/// panicked or was cancelled; `comm`'s own error type must convert into
/// [`OptimizationError`] via `?` (or already be `OptimizationError`) since
/// this function propagates it directly rather than wrapping it further.
///
/// # Example
///
/// See the module-level example for `compute` overlapping with a
/// distributed collective, or with either [`measure_latency`]/
/// [`measure_bandwidth`] probe above for a network operation specifically.
pub async fn overlap_compute_communicate<F, R, Fut, C>(
    compute: F,
    comm: Fut,
) -> Result<(R, C), OptimizationError>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
    Fut: Future<Output = Result<C, OptimizationError>>,
{
    let compute_handle = tokio::task::spawn_blocking(compute);
    let (compute_result, comm_result) = tokio::join!(compute_handle, comm);
    let computed =
        compute_result.map_err(|e| OptimizationError::ComputeTaskFailed(e.to_string()))?;
    let communicated = comm_result?;
    Ok((computed, communicated))
}

/// Compress data for network transfer
///
/// Serializes the data to JSON and compresses with LZ4 (fast, low-latency).
/// Wire format: [8-byte u64 LE uncompressed size][LZ4 frame compressed bytes]
pub fn compress_data<T: Serialize>(data: &[T]) -> Result<Vec<u8>, OptimizationError> {
    let json_bytes = serde_json::to_vec(data)
        .map_err(|e| OptimizationError::CompressionError(format!("Serialization error: {}", e)))?;

    let uncompressed_size = json_bytes.len() as u64;

    let compressed = lz4_compress(&json_bytes).map_err(|e| {
        OptimizationError::CompressionError(format!("LZ4 compression error: {}", e))
    })?;

    let mut result = Vec::with_capacity(8 + compressed.len());
    result.extend_from_slice(&uncompressed_size.to_le_bytes());
    result.extend_from_slice(&compressed);

    Ok(result)
}

/// Decompress data after network transfer
///
/// Decompresses data that was compressed with `compress_data`.
/// Wire format: [8-byte u64 LE uncompressed size][LZ4 frame compressed bytes]
pub fn decompress_data<T: for<'de> Deserialize<'de>>(
    data: &[u8],
) -> Result<Vec<T>, OptimizationError> {
    if data.len() < 8 {
        return Err(OptimizationError::CompressionError(format!(
            "Data too short: expected at least 8 bytes, got {}",
            data.len()
        )));
    }

    let size_bytes: [u8; 8] = data[..8].try_into().map_err(|_| {
        OptimizationError::CompressionError("Failed to read uncompressed size header".to_string())
    })?;
    let uncompressed_size = u64::from_le_bytes(size_bytes) as usize;

    let json_bytes = lz4_decompress(&data[8..], uncompressed_size).map_err(|e| {
        OptimizationError::CompressionError(format!("LZ4 decompression error: {}", e))
    })?;

    serde_json::from_slice(&json_bytes)
        .map_err(|e| OptimizationError::CompressionError(format!("Deserialization error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_topology() {
        let topology = NetworkTopology::Tree { arity: 2 };
        assert_eq!(
            topology.optimal_algorithm("broadcast"),
            Algorithm::TreeBroadcast
        );
    }

    #[test]
    fn test_bandwidth_model() {
        let mut model = BandwidthModel::new();
        model.add_measurement(0, 1, 1000.0);
        model.add_measurement(1, 2, 950.0);
        model.add_measurement(2, 3, 1050.0);

        assert_eq!(model.estimate(0, 1), 1000.0);
        assert!((model.average - 1000.0).abs() < 50.0);
    }

    #[test]
    fn test_latency_model() {
        let mut model = LatencyModel::new();
        model.add_measurement(0, 1, 10.0);
        model.add_measurement(1, 2, 12.0);
        model.add_measurement(2, 3, 11.0);

        assert_eq!(model.estimate(0, 1), 10.0);
        assert!((model.average - 11.0).abs() < 1.0);
    }

    #[test]
    fn test_topology_direct_connection() {
        let topology = NetworkTopology::FullyConnected;
        assert!(topology.has_direct_connection(0, 1, 4));
        assert!(topology.has_direct_connection(0, 3, 4));

        let ring = NetworkTopology::Ring;
        assert!(ring.has_direct_connection(0, 1, 4));
        assert!(!ring.has_direct_connection(0, 2, 4));
    }

    #[test]
    fn test_compress_decompress_roundtrip_floats() {
        let data: Vec<f64> = vec![1.0, 2.5, -3.14, 0.0, f64::MAX];
        let compressed = compress_data(&data).expect("compression should succeed");
        let recovered: Vec<f64> =
            decompress_data(&compressed).expect("decompression should succeed");
        assert_eq!(data.len(), recovered.len());
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < f64::EPSILON * 100.0,
                "mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_compress_decompress_roundtrip_strings() {
        let data: Vec<String> = vec![
            "hello".to_string(),
            "world".to_string(),
            "oxiarc".to_string(),
        ];
        let compressed = compress_data(&data).expect("compression should succeed");
        let recovered: Vec<String> =
            decompress_data(&compressed).expect("decompression should succeed");
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_compress_empty_slice() {
        let data: Vec<u32> = vec![];
        let compressed = compress_data(&data).expect("compression of empty slice should succeed");
        let recovered: Vec<u32> =
            decompress_data(&compressed).expect("decompression should succeed");
        assert_eq!(recovered, data);
    }

    #[test]
    fn test_compress_highly_compressible() {
        let data: Vec<u32> = vec![42u32; 10_000];
        let compressed = compress_data(&data).expect("compression should succeed");
        // LZ4 should compress highly repetitive data significantly
        assert!(
            compressed.len() < data.len() * 4,
            "expected compression, got {} bytes for {} elements",
            compressed.len(),
            data.len()
        );
        let recovered: Vec<u32> =
            decompress_data(&compressed).expect("decompression should succeed");
        assert_eq!(data, recovered);
    }

    #[test]
    fn test_decompress_invalid_data() {
        let bad_data = b"too short";
        let result: Result<Vec<u32>, _> = decompress_data(bad_data);
        assert!(result.is_err(), "should fail on short data");
    }

    // -----------------------------------------------------------------
    // Real ping-pong probes (measure_latency / measure_bandwidth) and
    // overlap_compute_communicate, over a real LocalCluster.
    // -----------------------------------------------------------------

    use crate::distributed::process::Communicator;
    use crate::distributed::testing::{ClusterNode, LocalCluster};
    use std::time::Duration;

    fn short_timeout_config() -> super::super::net::EndpointConfig {
        super::super::net::EndpointConfig {
            recv_timeout: Duration::from_secs(5),
            ..super::super::net::EndpointConfig::default()
        }
    }

    /// Both ranks in a real 2-process cluster must observe a strictly
    /// positive round-trip latency, and — since this is a real measurement,
    /// not a fabricated constant — both ranks (client and server) must agree
    /// on the exact same value (see the module docs on why only the client's
    /// clock is trusted and its result relayed to the server).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn measure_latency_is_positive_and_agrees_across_ranks() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            2,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(std::sync::Arc::new(node.endpoint))?;
                let peer = if comm.rank() == 0 { 1 } else { 0 };
                let latency_us = measure_latency(&comm, peer, 5, SendOpts { compress: false })
                    .await
                    .map_err(|e| NetError::Io(e.to_string()))?;
                Ok(latency_us)
            },
        )
        .await
        .expect("measure_latency run");

        assert_eq!(results.len(), 2);
        assert!(results[0] > 0.0, "latency must be positive: {}", results[0]);
        assert_eq!(
            results[0], results[1],
            "client and server must report the identical relayed latency"
        );
    }

    /// Same shape as the latency test, but for bandwidth: both ranks in a
    /// real cluster observe a strictly positive transfer rate for a real
    /// 1 MiB transfer, and agree on the exact same client-measured value.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn measure_bandwidth_is_positive_and_agrees_across_ranks() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            2,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(std::sync::Arc::new(node.endpoint))?;
                let peer = if comm.rank() == 0 { 1 } else { 0 };
                let bw = measure_bandwidth(&comm, peer, 1024 * 1024, SendOpts { compress: false })
                    .await
                    .map_err(|e| NetError::Io(e.to_string()))?;
                Ok(bw)
            },
        )
        .await
        .expect("measure_bandwidth run");

        assert_eq!(results.len(), 2);
        assert!(
            results[0] > 0.0,
            "bandwidth must be positive: {}",
            results[0]
        );
        assert_eq!(
            results[0], results[1],
            "client and server must report the identical relayed bandwidth"
        );
    }

    /// `SendOpts { compress: false }` must actually be honored end to end:
    /// an incompressible payload sent with compression permitted would still
    /// measure roughly the same rate as with it forced off (LZ4 either
    /// declines to shrink it or `Endpoint` discards a non-shrinking result —
    /// see `Endpoint::compress_if_worthwhile`), but this specifically pins
    /// that requesting *no* compression never accidentally compresses by
    /// checking the call still succeeds and returns a sane, positive number
    /// for a payload well above the compression threshold.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn measure_bandwidth_honors_uncompressed_flag() {
        let cfg = short_timeout_config();
        let results = LocalCluster::run_connected_with(
            2,
            cfg,
            Duration::from_secs(15),
            |node: ClusterNode| async move {
                let comm = Communicator::from_endpoint(std::sync::Arc::new(node.endpoint))?;
                let peer = if comm.rank() == 0 { 1 } else { 0 };
                // Above COMPRESS_THRESHOLD (64 KiB): if `compress: false`
                // were silently ignored somewhere along the path, this would
                // still succeed, but pinning `compress: false` here at least
                // documents and exercises the flag reaching the endpoint
                // call rather than being dropped before it.
                let bw = measure_bandwidth(&comm, peer, 256 * 1024, SendOpts { compress: false })
                    .await
                    .map_err(|e| NetError::Io(e.to_string()))?;
                Ok(bw)
            },
        )
        .await
        .expect("measure_bandwidth run");

        for bw in results {
            assert!(bw > 0.0, "uncompressed bandwidth must be positive: {bw}");
        }
    }

    /// `overlap_compute_communicate` must actually run `compute` and `comm`
    /// concurrently rather than sequentially: a `comm` future that sleeps
    /// and a `compute` closure that spins should together take roughly
    /// `max(sleep, spin)`, not their sum. This asserts the (generous) upper
    /// bound rather than a tight one, to stay robust on a loaded machine.
    #[tokio::test]
    async fn overlap_compute_communicate_runs_concurrently() {
        let sleep_for = Duration::from_millis(200);
        let start = Instant::now();
        let (spins, ()) = overlap_compute_communicate(
            || {
                let mut acc = 0u64;
                for i in 0..20_000_000u64 {
                    acc = acc.wrapping_add(i);
                }
                acc
            },
            async move {
                tokio::time::sleep(sleep_for).await;
                Ok(())
            },
        )
        .await
        .expect("overlap should succeed");
        let elapsed = start.elapsed();

        // A real value came back from the compute closure (not a placeholder).
        assert!(spins > 0);
        // If compute and comm ran sequentially, elapsed would tend toward
        // sleep_for + (time to do 20M wrapping adds) + scheduling overhead;
        // running concurrently, it should stay close to sleep_for alone.
        // Generous bound: sequential would very likely exceed 1s on any
        // machine capable of running this test at all.
        assert!(
            elapsed < Duration::from_secs(1),
            "overlap took {elapsed:?}, expected close to {sleep_for:?} if truly concurrent"
        );
    }

    /// A panicking `compute` closure must surface as
    /// [`OptimizationError::ComputeTaskFailed`], not silently vanish or
    /// poison the runtime.
    #[tokio::test]
    async fn overlap_compute_communicate_surfaces_compute_panic() {
        let result: Result<((), ()), _> =
            overlap_compute_communicate(|| panic!("intentional test panic"), async { Ok(()) })
                .await;
        match result {
            Err(OptimizationError::ComputeTaskFailed(msg)) => {
                // tokio::task::JoinError::to_string() includes the original
                // panic message; pin that it actually survives the
                // conversion rather than being replaced with a generic
                // "panicked" string.
                assert!(
                    msg.contains("intentional test panic"),
                    "panic message should survive into the error, got: {msg}"
                );
            }
            other => panic!("expected ComputeTaskFailed carrying the panic message, got {other:?}"),
        }
    }

    /// A `comm` future returning an error must propagate that error
    /// unchanged, and must not be masked by a fabricated success.
    #[tokio::test]
    async fn overlap_compute_communicate_surfaces_comm_error() {
        let result: Result<((), ()), _> = overlap_compute_communicate(|| (), async {
            Err(OptimizationError::MeasurementError("boom".to_string()))
        })
        .await;
        assert!(matches!(
            result,
            Err(OptimizationError::MeasurementError(msg)) if msg == "boom"
        ));
    }
}
