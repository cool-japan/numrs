//! Basic Distributed Computing Example for NumRS2
//!
//! Demonstrates the fundamental building blocks of distributed computing in
//! NumRS2: process/communicator setup, a block-distributed
//! [`DistributedArray`], global/local index conversion, ghost-cell boundary
//! exchange with rank neighbors, and barriers.
//!
//! # Running
//!
//! ```bash
//! cargo run --example distributed_basics --features distributed
//! ```
//!
//! That single command runs 4 ranks in *one* process over
//! [`LocalCluster`] — real loopback TCP, framing and all, just without the
//! multi-terminal dance a real cluster needs. For an actual multi-host run,
//! use [`numrs2::distributed::process::init`] instead, which reads its
//! configuration from `NUMRS2_RANK`/`NUMRS2_SIZE`/`NUMRS2_MASTER_ADDR` (see
//! that function's docs for the full contract):
//!
//! Terminal 1 (rank 0 of 2, runs the rendezvous master):
//! ```bash
//! NUMRS2_RANK=0 NUMRS2_SIZE=2 NUMRS2_MASTER_ADDR=127.0.0.1:5000 cargo run --example distributed_basics --features distributed
//! ```
//!
//! Terminal 2 (rank 1 of 2):
//! ```bash
//! NUMRS2_RANK=1 NUMRS2_SIZE=2 NUMRS2_MASTER_ADDR=127.0.0.1:5000 cargo run --example distributed_basics --features distributed
//! ```

#[cfg(feature = "distributed")]
use numrs2::distributed::net::NetError;
#[cfg(feature = "distributed")]
use numrs2::distributed::prelude::*;
#[cfg(feature = "distributed")]
use std::sync::Arc;

#[cfg(feature = "distributed")]
const WORLD_SIZE: u32 = 4;

#[cfg(feature = "distributed")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NumRS2 Distributed Computing - Basic Example");
    println!("============================================\n");
    println!(
        "Running {WORLD_SIZE} ranks in one process via LocalCluster (real loopback TCP\n\
         transport, no multi-terminal launch needed — see this file's header for a real\n\
         multi-host run).\n"
    );

    // Every rank's log lines are collected and printed after the run
    // completes (rather than printed live from concurrent tasks), so
    // output from four ranks racing each other doesn't interleave mid-line.
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

    println!("\nExample completed successfully.");
    Ok(())
}

/// One rank's share of the demo, run inside [`LocalCluster::run_connected`].
#[cfg(feature = "distributed")]
async fn run_rank(node: ClusterNode) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut log = Vec::new();
    let world = Communicator::from_endpoint(Arc::new(node.endpoint))?;

    let rank = world.rank();
    let size = world.size();
    log.push(format!(
        "[rank {rank}] initialized ({rank} of {size}); hostname={}, addr={}, root={}",
        world.process_info().hostname,
        world.process_info().addr,
        world.is_root()
    ));

    // Create a distributed array: each rank contributes an equal block.
    let local_len = 10;
    let local_data: Vec<f64> = (0..local_len)
        .map(|i| (rank * local_len + i) as f64)
        .collect();
    let global_size = size * local_len;
    let mut dist_array =
        DistributedArray::from_local(local_data, DistributionStrategy::Block, global_size, &world)?;
    log.push(format!(
        "[rank {rank}] distributed array: global_size={}, local_size={}",
        dist_array.global_size(),
        dist_array.local_size()
    ));

    // Global <-> local index conversion. The midpoint is in range for any
    // world size, unlike a hardcoded index that only happens to fit one.
    if world.is_root() {
        let probe = global_size / 2;
        match dist_array.global_to_local(&GlobalIndex::new(probe))? {
            Some(local_idx) => log.push(format!(
                "[rank {rank}] global index {probe} -> local index {} (owned here)",
                local_idx.index()
            )),
            None => log.push(format!(
                "[rank {rank}] global index {probe} -> owned by another rank"
            )),
        }
        let global_back = dist_array.local_to_global(&LocalIndex::new(0))?;
        log.push(format!(
            "[rank {rank}] local index 0 -> global index {}",
            global_back.index()
        ));
    }

    barrier(&world).await?;

    // Ghost-cell boundary exchange: a real point-to-point transfer with the
    // immediate rank neighbors (see `DistributedArray::sync_ghost_cells`).
    // Rank 0 has no left neighbor and the last rank has no right neighbor,
    // so those ends stay empty — everyone else sees real neighbor data.
    dist_array.init_ghost_cells(2);
    dist_array.sync_ghost_cells().await?;
    if let Some(ghosts) = dist_array.ghost_cells() {
        log.push(format!(
            "[rank {rank}] ghost cells: left={:?}, right={:?}",
            ghosts.left(),
            ghosts.right()
        ));
    }

    barrier(&world).await?;
    if world.is_root() {
        log.push(format!(
            "[rank {rank}] process group: size={}, ranks={:?}",
            world.group().size(),
            world.group().ranks
        ));
    }

    Ok(log)
}

#[cfg(not(feature = "distributed"))]
fn main() {
    eprintln!("This example requires the 'distributed' feature.");
    eprintln!("Run with: cargo run --example distributed_basics --features distributed");
    std::process::exit(1);
}
