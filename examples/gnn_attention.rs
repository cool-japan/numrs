//! Graph Attention Network (GAT) Example
//!
//! This example demonstrates attention mechanisms in graph neural networks using GAT.
//! We show how attention weights learn to focus on important neighbors.
//!
//! # Task
//!
//! Given a social network where:
//! - Nodes represent users
//! - Edges represent friendships
//! - Node features represent user interests/activity
//! - Goal: Learn user representations that weight important connections
//!
//! # Method
//!
//! We use GAT with multi-head attention:
//! 1. Compute attention coefficients for each neighbor
//! 2. Aggregate neighbor features with learned attention weights
//! 3. Use multiple attention heads to capture different aspects
//!
//! # Reference
//!
//! Veličković et al. (2018) - "Graph Attention Networks" (ICLR)

use numrs2::new_modules::nn::graph::*;
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Graph Attention Network (GAT) Example ===\n");

    // Create a social network
    // 5 users with different interest profiles
    let num_users = 5;

    // Friendship network (undirected)
    let friendships = vec![
        // User 0: Central hub (friends with everyone)
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (0, 3),
        (3, 0),
        (0, 4),
        (4, 0),
        // Some inter-user friendships
        (1, 2),
        (2, 1),
        (3, 4),
        (4, 3),
    ];

    println!("Social Network:");
    println!("  Users: {}", num_users);
    println!("  Friendships: {} (undirected)", friendships.len() / 2);
    println!("  User 0 is a central hub (connected to all others)");
    println!();

    // Create sparse adjacency
    let social_graph = SparseAdjacency::<f64>::from_edges(num_users, &friendships)?;

    // Analyze network structure
    let degrees = social_graph.degrees();
    println!("Network Structure:");
    for (i, &deg) in degrees.iter().enumerate() {
        println!("  User {}: {} friends", i, deg);
    }
    println!();

    // User interest features (6-dimensional)
    // Features represent interests: [sports, tech, music, art, food, travel]
    let user_features = Array2::from_shape_fn((num_users, 6), |(i, j)| {
        match i {
            0 => [0.8, 0.7, 0.5, 0.6, 0.7, 0.9][j], // Hub: diverse interests
            1 => [0.9, 0.3, 0.2, 0.1, 0.4, 0.3][j], // Sports fan
            2 => [0.2, 0.9, 0.8, 0.7, 0.3, 0.5][j], // Tech enthusiast
            3 => [0.1, 0.4, 0.9, 0.9, 0.2, 0.6][j], // Art & music lover
            4 => [0.3, 0.2, 0.3, 0.2, 0.9, 0.9][j], // Food & travel
            _ => 0.0,
        }
    });

    println!("User Interest Profiles:");
    let interests = ["Sports", "Tech", "Music", "Art", "Food", "Travel"];
    println!("{:-<70}", "");
    print!("{:<8}", "User");
    for interest in &interests {
        print!("{:^10}", interest);
    }
    println!();
    println!("{:-<70}", "");

    for i in 0..num_users {
        print!("{:<8}", i);
        for j in 0..6 {
            print!("{:^10.2}", user_features[[i, j]]);
        }
        println!();
    }
    println!("{:-<70}", "");
    println!();

    // Build GAT layer
    let input_dim = 6;
    let output_dim = 4;
    let num_heads = 2;
    let concat = true; // Concatenate attention heads
    let alpha = 0.2; // LeakyReLU slope

    println!("GAT Configuration:");
    println!("  Input features: {}", input_dim);
    println!("  Output features per head: {}", output_dim);
    println!("  Number of attention heads: {}", num_heads);
    println!(
        "  Head combination: {}",
        if concat { "Concatenate" } else { "Average" }
    );
    println!("  LeakyReLU slope: {}", alpha);
    println!(
        "  Total output dimension: {}",
        if concat {
            output_dim * num_heads
        } else {
            output_dim
        }
    );
    println!();

    let gat_layer = GatLayer::new(input_dim, output_dim, num_heads, concat, alpha)?;

    // Forward pass
    println!("Running GAT forward pass...");
    let output = gat_layer.forward(&social_graph, &user_features.view())?;
    println!("  Output shape: {:?}", output.shape());
    println!(
        "  Output dimension: {} ({}×{} heads concatenated)",
        output.ncols(),
        num_heads,
        output_dim
    );
    println!();

    // Analyze output representations
    println!("Learned User Representations:");
    println!("{:-<50}", "");
    for i in 0..num_users {
        print!("  User {}: [", i);
        for j in 0..4.min(output.ncols()) {
            print!("{:7.3}", output[[i, j]]);
            if j < 3 && j < output.ncols() - 1 {
                print!(", ");
            }
        }
        if output.ncols() > 4 {
            print!(", ...");
        }
        println!("]");
    }
    println!("{:-<50}", "");
    println!();

    // Compute similarity between users
    println!("User Representation Similarity:");
    println!("{:-<40}", "");
    println!("{:<15} {:>10}", "User Pair", "Cosine Sim");
    println!("{:-<40}", "");

    for i in 0..num_users {
        for j in (i + 1)..num_users {
            let mut dot = 0.0;
            let mut norm_i = 0.0;
            let mut norm_j = 0.0;

            for k in 0..output.ncols() {
                dot += output[[i, k]] * output[[j, k]];
                norm_i += output[[i, k]] * output[[i, k]];
                norm_j += output[[j, k]] * output[[j, k]];
            }

            let cosine_sim = dot / (norm_i.sqrt() * norm_j.sqrt());
            let connected = friendships
                .iter()
                .any(|&(a, b)| (a == i && b == j) || (a == j && b == i));

            println!(
                "{:<15} {:>10.4} {}",
                format!("User {} - {}", i, j),
                cosine_sim,
                if connected { "✓ connected" } else { "" }
            );
        }
    }
    println!("{:-<40}", "");
    println!();

    // Explain attention mechanism
    println!("Attention Mechanism Explanation:");
    println!("  1. For each edge (i,j), compute attention logit:");
    println!("     e_ij = LeakyReLU(a^T [Wh_i || Wh_j])");
    println!("  2. Normalize with softmax over neighbors:");
    println!("     α_ij = exp(e_ij) / Σ_k exp(e_ik)");
    println!("  3. Aggregate with attention weights:");
    println!("     h_i' = Σ_j α_ij W h_j");
    println!();

    // Multi-head attention benefits
    println!("Multi-Head Attention Benefits:");
    println!("  - Head 1 might focus on one type of relationship");
    println!("  - Head 2 might capture different structural patterns");
    println!("  - Concatenation combines multiple perspectives");
    println!("  - Provides richer representations than single attention");
    println!();

    // Demonstrate attention focusing
    println!("Attention Properties:");
    println!("  ✓ Self-attention: Can attend to own features");
    println!("  ✓ Neighbor weighting: Important neighbors get higher weights");
    println!("  ✓ Permutation invariant: Order of neighbors doesn't matter");
    println!("  ✓ Inductive: Can process new nodes without retraining");
    println!();

    // Compare with mean aggregation
    println!("Comparison with Mean Aggregation:");
    let mean_agg = mean_aggregation(&social_graph, &user_features.view())?;
    println!("  Mean aggregation shape: {:?}", mean_agg.shape());
    println!("  GAT learns adaptive weights, mean uses uniform 1/|N(i)|");
    println!("  GAT can focus on relevant neighbors for the task");
    println!();

    // Show graph pooling for graph classification
    println!("Graph-Level Pooling (for graph classification):");
    let graph_mean = global_mean_pool(&output.view())?;
    let graph_max = global_max_pool(&output.view())?;
    println!("  Global mean: shape {:?}", graph_mean.shape());
    println!("  Global max: shape {:?}", graph_max.shape());
    println!("  These representations summarize entire social network");
    println!();

    println!("=== Example completed successfully! ===");
    println!("\nNext steps:");
    println!("  1. Train GAT on real node classification tasks");
    println!("  2. Visualize learned attention weights");
    println!("  3. Experiment with different numbers of heads");
    println!("  4. Combine GAT with other GNN layers (GCN, GraphSAGE)");
    println!("  5. Apply to link prediction or graph generation");

    Ok(())
}
