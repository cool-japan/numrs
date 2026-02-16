//! Basic Distributed Computing Example for NumRS2
//!
//! This example demonstrates the fundamental concepts of distributed computing
//! in NumRS2, including process initialization, distributed arrays, and collective
//! operations.
//!
//! # Running the Example
//!
//! To run this example with multiple processes, set environment variables:
//!
//! Terminal 1 (Process 0 - Master):
//! ```bash
//! NUMRS2_RANK=0 NUMRS2_SIZE=2 NUMRS2_MASTER_ADDR=127.0.0.1:5000 cargo run --example distributed_basics --features distributed
//! ```
//!
//! Terminal 2 (Process 1):
//! ```bash
//! NUMRS2_RANK=1 NUMRS2_SIZE=2 NUMRS2_MASTER_ADDR=127.0.0.1:5000 cargo run --example distributed_basics --features distributed
//! ```

#[cfg(feature = "distributed")]
use numrs2::distributed::prelude::*;

#[cfg(feature = "distributed")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NumRS2 Distributed Computing - Basic Example");
    println!("============================================\n");

    // Initialize distributed environment
    println!("Initializing distributed environment...");
    let world = init().await?;

    let rank = world.rank();
    let size = world.size();

    println!("✓ Process {} of {} initialized successfully", rank, size);
    println!("  Hostname: {}", world.process_info().hostname);
    println!("  Address: {}\n", world.process_info().addr);

    // Demonstrate process information
    println!("Process Information:");
    println!("  Rank: {}", rank);
    println!("  Size: {}", size);
    println!("  Is root: {}\n", world.is_root());

    // Create local data
    println!("Creating local data...");
    let local_data: Vec<f64> = (0..10).map(|i| (rank * 10 + i) as f64).collect();
    println!("  Local data (rank {}): {:?}\n", rank, &local_data[..5]);

    // Demonstrate distributed array
    println!("Creating distributed array...");
    let global_size = size * 10;
    let dist_array = DistributedArray::from_local(
        local_data.clone(),
        DistributionStrategy::Block,
        global_size,
        &world,
    )?;

    println!("✓ Distributed array created");
    println!("  Global size: {}", dist_array.global_size());
    println!(
        "  Local size (rank {}): {}\n",
        rank,
        dist_array.local_size()
    );

    // Demonstrate index conversion
    if rank == 0 {
        println!("Index Conversion Examples:");
        let global_idx = GlobalIndex::new(15);
        println!("  Global index 15:");

        match dist_array.global_to_local(&global_idx)? {
            Some(local_idx) => {
                println!("    → Local index: {} (on this process)", local_idx.index());
            }
            None => {
                println!("    → Not owned by this process");
            }
        }

        let local_idx = LocalIndex::new(5);
        let global = dist_array.local_to_global(&local_idx)?;
        println!("  Local index 5 → Global index: {}\n", global.index());
    }

    // Barrier synchronization
    println!("Synchronizing at barrier...");
    barrier(&world).await?;
    println!("✓ All processes reached barrier\n");

    // Demonstrate process group information
    if world.is_root() {
        println!("Process Group Information:");
        println!("  Total processes: {}", world.group().size());
        println!("  Ranks: {:?}\n", world.group().ranks);
    }

    // Another barrier before finalization
    barrier(&world).await?;

    // Finalize distributed environment
    if world.is_root() {
        println!("\nFinalizing distributed environment...");
    }

    finalize(world).await?;

    if rank == 0 {
        println!("✓ Distributed environment finalized successfully");
        println!("\nExample completed successfully! 🎉");
    }

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    eprintln!("This example requires the 'distributed' feature.");
    eprintln!("Run with: cargo run --example distributed_basics --features distributed");
    std::process::exit(1);
}
