//! Comprehensive Distributed Computing Example for NumRS2
//!
//! Demonstrates the full breadth of the `distributed` feature:
//! - Distributed array operations (scatter, gather, allreduce, broadcast)
//! - Collective communication
//! - Real distributed linear algebra: [`DistributedMatrix`] + `matmul`,
//!   `distributed_qr` and `distributed_solve_spd` (Cholesky)
//! - Error handling patterns
//!
//! # Running
//!
//! ```bash
//! cargo run --example distributed_computing --features distributed
//! ```
//!
//! That single command runs 4 ranks in *one* process over [`LocalCluster`] —
//! real loopback TCP, framing and all, just without the multi-terminal dance
//! a real cluster needs. For an actual multi-host run, use
//! [`numrs2::distributed::process::init`] instead (`NUMRS2_RANK`/
//! `NUMRS2_SIZE`/`NUMRS2_MASTER_ADDR`) — see `distributed_basics.rs`'s header
//! for the exact multi-terminal invocation.

#[cfg(feature = "distributed")]
use numrs2::distributed::net::NetError;
#[cfg(feature = "distributed")]
use numrs2::distributed::prelude::*;
#[cfg(feature = "distributed")]
use scirs2_core::ndarray::Array2;
#[cfg(feature = "distributed")]
use std::sync::Arc;

#[cfg(feature = "distributed")]
const WORLD_SIZE: u32 = 4;

#[cfg(feature = "distributed")]
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("NumRS2 Distributed Computing - Comprehensive Example");
    println!("===================================================\n");
    println!("Running {WORLD_SIZE} ranks in one process via LocalCluster (real loopback TCP).\n");

    let per_rank_logs = LocalCluster::run_connected(WORLD_SIZE, |node: ClusterNode| async move {
        run_rank(node)
            .await
            .map_err(|e| NetError::Io(e.to_string()))
    })
    .await?;

    for log in per_rank_logs {
        for line in log {
            println!("{line}");
        }
    }

    println!("\n✓ All examples completed successfully!");
    Ok(())
}

#[cfg(feature = "distributed")]
async fn run_rank(
    node: ClusterNode,
) -> std::result::Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut log = Vec::new();

    // `world` (a `Communicator`) drives examples 1-4 and 6, over the real
    // collectives in `distributed::collective`; `transport` (an
    // `EndpointTransport` wrapping a clone of the same endpoint — cloning is
    // cheap, it shares one link set) drives example 5's `DistributedMatrix`
    // algorithms, which are written against the more general `DistTransport`
    // trait rather than against `Communicator` directly. The two never
    // collide on the wire: a `Communicator`'s collectives always run under
    // context 0, while every `DistTransport` algorithm allocates its own
    // context starting at 1 (see `distributed::linalg`'s module docs).
    let transport = EndpointTransport::new(node.endpoint.clone());
    let world = Communicator::from_endpoint(Arc::new(node.endpoint))?;

    example1_distributed_arrays(&world, &mut log).await?;
    example2_scatter_gather(&world, &mut log).await?;
    example3_allreduce(&world, &mut log).await?;
    example4_broadcast(&world, &mut log).await?;
    example5_distributed_linalg(&transport, &mut log).await?;
    example6_error_handling(&world, &mut log).await?;

    Ok(log)
}

#[cfg(feature = "distributed")]
async fn example1_distributed_arrays(
    world: &Communicator,
    log: &mut Vec<String>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        log.push("=== Example 1: Distributed Array Creation ===".to_string());
    }
    barrier(world).await?;

    let rank = world.rank();
    let size = world.size();

    let local_size = 10;
    let local_data: Vec<f64> = (0..local_size)
        .map(|i| (rank * local_size + i) as f64)
        .collect();

    log.push(format!(
        "[rank {rank}] local data: [{:.1}, {:.1}, ..., {:.1}]",
        local_data[0],
        local_data[1],
        local_data[local_size - 1]
    ));

    let global_size = size * local_size;
    let dist_array =
        DistributedArray::from_local(local_data, DistributionStrategy::Block, global_size, world)?;

    log.push(format!(
        "[rank {rank}] distributed array: global_size={}, local_size={}",
        dist_array.global_size(),
        dist_array.local_size()
    ));

    barrier(world).await?;
    Ok(())
}

#[cfg(feature = "distributed")]
async fn example2_scatter_gather(
    world: &Communicator,
    log: &mut Vec<String>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        log.push("=== Example 2: Scatter and Gather Operations ===".to_string());
    }
    barrier(world).await?;

    let rank = world.rank();
    let size = world.size();

    let scatter_data = if world.is_root() {
        let mut data = Vec::new();
        for i in 0..size {
            for j in 0..5 {
                data.push((i * 100 + j) as f64);
            }
        }
        log.push(format!(
            "[rank 0] prepared {} elements to scatter",
            data.len()
        ));
        data
    } else {
        vec![]
    };

    let local_chunk = scatter(&scatter_data, 0, world).await?;
    log.push(format!(
        "[rank {rank}] received scattered data: {local_chunk:?}"
    ));

    barrier(world).await?;

    let modified_chunk: Vec<f64> = local_chunk.iter().map(|x| x * 2.0).collect();
    let gathered_data = gather(&modified_chunk, 0, world).await?;

    if world.is_root() {
        log.push(format!(
            "[rank 0] gathered {} elements: {:?}",
            gathered_data.len(),
            gathered_data
        ));
    }

    barrier(world).await?;
    Ok(())
}

#[cfg(feature = "distributed")]
async fn example3_allreduce(
    world: &Communicator,
    log: &mut Vec<String>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        log.push("=== Example 3: Allreduce Operations ===".to_string());
    }
    barrier(world).await?;

    let rank = world.rank();
    let local_data = vec![(rank + 1) as f64; 5];

    let sum_result = allreduce(&local_data, ReduceOp::Sum, world).await?;
    let max_result = allreduce(&local_data, ReduceOp::Max, world).await?;
    let min_result = allreduce(&local_data, ReduceOp::Min, world).await?;

    log.push(format!(
        "[rank {rank}] local={local_data:?} -> sum={sum_result:?}, max={max_result:?}, min={min_result:?}"
    ));

    barrier(world).await?;
    Ok(())
}

#[cfg(feature = "distributed")]
async fn example4_broadcast(
    world: &Communicator,
    log: &mut Vec<String>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        log.push("=== Example 4: Broadcast Operations ===".to_string());
    }
    barrier(world).await?;

    let rank = world.rank();
    let mut broadcast_data = if world.is_root() {
        vec![2.5, 2.71, 1.41, 1.73, 2.23]
    } else {
        vec![0.0; 5]
    };

    broadcast(&mut broadcast_data, 0, world).await?;
    log.push(format!(
        "[rank {rank}] received broadcast: {broadcast_data:?}"
    ));

    barrier(world).await?;
    Ok(())
}

/// Real distributed linear algebra over [`DistributedMatrix`] — the
/// working surface `distributed_qr`/`distributed_svd`/`distributed_solve`/
/// `block_cholesky` actually live on (the `DistributedArray`-surface
/// functions of the same names in [`numrs2::distributed::linalg`] are
/// permanently `NotImplemented` shims naming this replacement — see that
/// module's docs).
#[cfg(feature = "distributed")]
async fn example5_distributed_linalg(
    transport: &EndpointTransport,
    log: &mut Vec<String>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let rank = transport.rank();
    let world_size = transport.world_size();
    if rank == 0 {
        log.push("=== Example 5: Distributed Linear Algebra (DistributedMatrix) ===".to_string());
    }

    // --- matmul: every rank's row-block of A, times a replicated 2*I ---
    let rows = world_size as usize * 4;
    let cols = 4;
    let a_global = Array2::from_shape_fn((rows, cols), |(i, j)| (i + 2 * j + 1) as f64);
    let two_eye = Array2::from_shape_fn((cols, cols), |(i, j)| if i == j { 2.0 } else { 0.0 });

    let a = DistributedMatrix::from_global(Layout::RowBlock, &a_global.view(), rank, world_size)?;
    let b = DistributedMatrix::from_global(Layout::RowBlock, &two_eye.view(), rank, world_size)?;
    let c = matmul(&a, &b, transport).await?;
    let matmul_ok = c
        .local_view()
        .iter()
        .zip(a.local_view().iter())
        .all(|(actual, original)| (actual - 2.0 * original).abs() < 1e-9);
    log.push(format!(
        "[rank {rank}] matmul: A * (2I) == 2A ? {matmul_ok}"
    ));

    // --- distributed_qr: factor A = QR, reconstruct at root, check the residual ---
    let (q, r) = distributed_qr(&a, transport).await?;
    if let Some(full_q) = q.gather_to_root(transport, 0).await? {
        let reconstructed = full_q.dot(&r);
        let residual = (&reconstructed - &a_global)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        log.push(format!(
            "[rank {rank}] distributed_qr: ||A - QR|| = {residual:.3e}"
        ));
    }

    // --- distributed_solve_spd (Cholesky): a fixed tridiagonal SPD system ---
    let n = 4usize;
    let spd = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j {
            4.0
        } else if i.abs_diff(j) == 1 {
            1.0
        } else {
            0.0
        }
    });
    let rhs = vec![1.0_f64, 2.0, 3.0, 4.0];
    let spd_dist = DistributedMatrix::from_global(
        Layout::ColBlockCyclic { panel_width: 1 },
        &spd.view(),
        rank,
        world_size,
    )?;
    let x = distributed_solve_spd(&spd_dist, &rhs, transport).await?;
    if rank == 0 {
        let residual = (0..n)
            .map(|i| {
                let ax_i: f64 = (0..n).map(|j| spd[[i, j]] * x[j]).sum();
                (ax_i - rhs[i]).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        log.push(format!(
            "[rank {rank}] distributed_solve_spd: x={x:?}, ||Ax - b|| = {residual:.3e}"
        ));
    }

    Ok(())
}

#[cfg(feature = "distributed")]
async fn example6_error_handling(
    world: &Communicator,
    log: &mut Vec<String>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        log.push("=== Example 6: Error Handling Patterns ===".to_string());
    }
    barrier(world).await?;

    let rank = world.rank();
    let safe_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    match allreduce(&safe_data, ReduceOp::Sum, world).await {
        Ok(result) => log.push(format!(
            "[rank {rank}] safe allreduce succeeded: {result:?}"
        )),
        Err(e) => {
            log.push(format!("[rank {rank}] error in allreduce: {e:?}"));
            return Err(e.into());
        }
    }

    barrier(world).await?;
    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    eprintln!("This example requires the 'distributed' feature.");
    eprintln!("Run with: cargo run --example distributed_computing --features distributed");
    std::process::exit(1);
}
