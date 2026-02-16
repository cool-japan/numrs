//! Comprehensive Distributed Computing Example for NumRS2
//!
//! This example demonstrates advanced distributed computing patterns including:
//! - Distributed array operations (scatter, gather, allreduce)
//! - Collective communications
//! - Distributed linear algebra
//! - Error handling and fault tolerance patterns
//!
//! # Running the Example
//!
//! To run this example with multiple processes, set environment variables:
//!
//! Terminal 1 (Process 0 - Master):
//! ```bash
//! NUMRS2_RANK=0 NUMRS2_SIZE=4 NUMRS2_MASTER_ADDR=127.0.0.1:5000 cargo run --example distributed_computing --features distributed
//! ```
//!
//! Terminal 2 (Process 1):
//! ```bash
//! NUMRS2_RANK=1 NUMRS2_SIZE=4 NUMRS2_MASTER_ADDR=127.0.0.1:5000 cargo run --example distributed_computing --features distributed
//! ```
//!
//! Terminal 3 (Process 2):
//! ```bash
//! NUMRS2_RANK=2 NUMRS2_SIZE=4 NUMRS2_MASTER_ADDR=127.0.0.1:5000 cargo run --example distributed_computing --features distributed
//! ```
//!
//! Terminal 4 (Process 3):
//! ```bash
//! NUMRS2_RANK=3 NUMRS2_SIZE=4 NUMRS2_MASTER_ADDR=127.0.0.1:5000 cargo run --example distributed_computing --features distributed
//! ```

#[cfg(feature = "distributed")]
use numrs2::distributed::prelude::*;
#[cfg(feature = "distributed")]
use numrs2::prelude::*;

#[cfg(feature = "distributed")]
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("NumRS2 Distributed Computing - Comprehensive Example");
    println!("===================================================\n");

    // Initialize distributed environment
    println!("Initializing distributed environment...");
    let world = init().await?;

    let rank = world.rank();
    let size = world.size();

    println!("✓ Process {} of {} initialized successfully", rank, size);
    println!("  Hostname: {}", world.process_info().hostname);
    println!("  Address: {}\n", world.process_info().addr);

    // Example 1: Distributed Array Creation and Basic Operations
    example1_distributed_arrays(&world).await?;

    // Example 2: Scatter and Gather Operations
    example2_scatter_gather(&world).await?;

    // Example 3: Allreduce Operations
    example3_allreduce(&world).await?;

    // Example 4: Broadcast Operations
    example4_broadcast(&world).await?;

    // Example 5: Distributed Matrix Operations
    example5_distributed_linalg(&world).await?;

    // Example 6: Error Handling Patterns
    example6_error_handling(&world).await?;

    // Finalize
    barrier(&world).await?;
    if world.is_root() {
        println!("\n✓ All examples completed successfully!");
        println!("Finalizing distributed environment...");
    }

    finalize(world).await?;

    if rank == 0 {
        println!("✓ Distributed environment finalized successfully");
    }

    Ok(())
}

#[cfg(feature = "distributed")]
async fn example1_distributed_arrays(
    world: &Communicator,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        println!("\n=== Example 1: Distributed Array Creation ===");
    }
    barrier(world).await?;

    let rank = world.rank();
    let size = world.size();

    // Each process creates local data
    let local_size = 10;
    let local_data: Vec<f64> = (0..local_size)
        .map(|i| (rank * local_size + i) as f64)
        .collect();

    println!(
        "[Rank {}] Created local data: [{:.1}, {:.1}, ..., {:.1}]",
        rank,
        local_data[0],
        local_data[1],
        local_data[local_size - 1]
    );

    // Create distributed array
    let global_size = size * local_size;
    let dist_array = DistributedArray::from_local(
        local_data.clone(),
        DistributionStrategy::Block,
        global_size,
        world,
    )?;

    println!(
        "[Rank {}] Distributed array: global_size={}, local_size={}",
        rank,
        dist_array.global_size(),
        dist_array.local_size()
    );

    barrier(world).await?;

    if world.is_root() {
        println!("✓ Example 1 completed\n");
    }

    Ok(())
}

#[cfg(feature = "distributed")]
async fn example2_scatter_gather(
    world: &Communicator,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        println!("=== Example 2: Scatter and Gather Operations ===");
    }
    barrier(world).await?;

    let rank = world.rank();
    let size = world.size();

    // Root process prepares data to scatter
    let scatter_data = if world.is_root() {
        let mut data = Vec::new();
        for i in 0..size {
            for j in 0..5 {
                data.push((i * 100 + j) as f64);
            }
        }
        println!("[Rank 0] Prepared {} elements to scatter", data.len());
        data
    } else {
        vec![]
    };

    // Scatter data from root to all processes
    let local_chunk = scatter(&scatter_data, 0, world).await?;

    println!("[Rank {}] Received scattered data: {:?}", rank, local_chunk);

    barrier(world).await?;

    // Each process modifies its local data
    let modified_chunk: Vec<f64> = local_chunk.iter().map(|x| x * 2.0).collect();

    println!("[Rank {}] Modified local data: {:?}", rank, modified_chunk);

    // Gather all modified data back to root
    let gathered_data = gather(&modified_chunk, 0, world).await?;

    if world.is_root() {
        println!(
            "[Rank 0] Gathered {} elements from all processes",
            gathered_data.len()
        );
        println!(
            "[Rank 0] First 10 elements: {:?}",
            &gathered_data[..10.min(gathered_data.len())]
        );
    }

    barrier(world).await?;

    if world.is_root() {
        println!("✓ Example 2 completed\n");
    }

    Ok(())
}

#[cfg(feature = "distributed")]
async fn example3_allreduce(
    world: &Communicator,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        println!("=== Example 3: Allreduce Operations ===");
    }
    barrier(world).await?;

    let rank = world.rank();

    // Each process has local data
    let local_data = vec![(rank + 1) as f64; 5];
    println!("[Rank {}] Local data: {:?}", rank, local_data);

    // Sum reduction across all processes
    let sum_result = allreduce(&local_data, ReduceOp::Sum, world).await?;
    println!("[Rank {}] Sum result: {:?}", rank, sum_result);

    // Max reduction across all processes
    let max_result = allreduce(&local_data, ReduceOp::Max, world).await?;
    println!("[Rank {}] Max result: {:?}", rank, max_result);

    // Min reduction across all processes
    let min_result = allreduce(&local_data, ReduceOp::Min, world).await?;
    println!("[Rank {}] Min result: {:?}", rank, min_result);

    barrier(world).await?;

    if world.is_root() {
        println!("✓ Example 3 completed\n");
    }

    Ok(())
}

#[cfg(feature = "distributed")]
async fn example4_broadcast(
    world: &Communicator,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        println!("=== Example 4: Broadcast Operations ===");
    }
    barrier(world).await?;

    let rank = world.rank();

    // Root process prepares data to broadcast
    let mut broadcast_data = if world.is_root() {
        let data = vec![2.5, 2.71, 1.41, 1.73, 2.23];
        println!("[Rank 0] Broadcasting data: {:?}", data);
        data
    } else {
        vec![0.0; 5]
    };

    // Broadcast from root to all processes
    broadcast(&mut broadcast_data, 0, world).await?;

    println!(
        "[Rank {}] Received broadcast data: {:?}",
        rank, broadcast_data
    );

    barrier(world).await?;

    if world.is_root() {
        println!("✓ Example 4 completed\n");
    }

    Ok(())
}

#[cfg(feature = "distributed")]
async fn example5_distributed_linalg(
    world: &Communicator,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        println!("=== Example 5: Distributed Linear Algebra ===");
    }
    barrier(world).await?;

    let rank = world.rank();
    let size = world.size();

    // Each process creates part of a distributed matrix
    let rows_per_proc = 4;
    let cols = 8;
    let local_matrix: Vec<f64> = (0..rows_per_proc * cols)
        .map(|i| (rank * rows_per_proc * cols + i) as f64 * 0.1)
        .collect();

    println!(
        "[Rank {}] Created local matrix chunk: {}x{}",
        rank, rows_per_proc, cols
    );

    // Compute local statistics
    let local_sum: f64 = local_matrix.iter().sum();
    let local_mean = local_sum / local_matrix.len() as f64;

    println!(
        "[Rank {}] Local statistics: sum={:.2}, mean={:.4}",
        rank, local_sum, local_mean
    );

    // Compute global statistics using allreduce
    let global_sum_vec = allreduce(&[local_sum], ReduceOp::Sum, world).await?;
    let global_sum = global_sum_vec[0];
    let total_elements = (size * rows_per_proc * cols) as f64;
    let global_mean = global_sum / total_elements;

    println!(
        "[Rank {}] Global statistics: sum={:.2}, mean={:.4}",
        rank, global_sum, global_mean
    );

    barrier(world).await?;

    if world.is_root() {
        println!("✓ Example 5 completed\n");
    }

    Ok(())
}

#[cfg(feature = "distributed")]
async fn example6_error_handling(
    world: &Communicator,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if world.is_root() {
        println!("=== Example 6: Error Handling Patterns ===");
    }
    barrier(world).await?;

    let rank = world.rank();

    // Example of safe operation with error handling
    let safe_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    match allreduce(&safe_data, ReduceOp::Sum, world).await {
        Ok(result) => {
            println!("[Rank {}] Safe allreduce succeeded: {:?}", rank, result);
        }
        Err(e) => {
            println!("[Rank {}] Error in allreduce: {:?}", rank, e);
            return Err(e.into());
        }
    }

    barrier(world).await?;

    if world.is_root() {
        println!("✓ Example 6 completed\n");
        println!("All error handling patterns executed successfully");
    }

    Ok(())
}

#[cfg(not(feature = "distributed"))]
fn main() {
    eprintln!("This example requires the 'distributed' feature.");
    eprintln!("Run with: cargo run --example distributed_computing --features distributed");
    std::process::exit(1);
}
