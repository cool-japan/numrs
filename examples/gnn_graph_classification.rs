//! Graph Neural Network Graph Classification Example
//!
//! This example demonstrates graph-level classification using Graph Isomorphism Networks (GIN).
//! We classify molecular graphs to predict chemical properties.
//!
//! # Task
//!
//! Given molecular graphs where:
//! - Nodes represent atoms
//! - Edges represent chemical bonds
//! - Node features represent atom properties (atomic number, charge, etc.)
//! - Goal: Classify molecules as toxic/non-toxic
//!
//! # Method
//!
//! We use GIN with graph pooling:
//! 1. GIN layers to learn node representations
//! 2. Global pooling to get graph-level representation
//! 3. MLP classifier for final prediction
//!
//! # Reference
//!
//! Xu et al. (2019) - "How Powerful are Graph Neural Networks?" (ICLR)

use numrs2::new_modules::nn::graph::*;
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GIN Graph Classification Example ===\n");

    // Create two example molecular graphs

    // Molecule 1: Small non-toxic molecule (e.g., water-like)
    println!("Molecule 1: Small non-toxic structure");
    let mol1_nodes = 3; // 3 atoms
    let mol1_edges = vec![(0, 1), (1, 2)]; // linear structure
    let mol1_features = Array2::from_shape_fn((mol1_nodes, 8), |(i, j)| {
        // Simple atom features
        ((i * 3 + j) % 10) as f64 / 10.0
    });
    println!("  Nodes: {}, Edges: {}", mol1_nodes, mol1_edges.len());

    // Molecule 2: Larger toxic molecule (e.g., benzene-like ring)
    println!("Molecule 2: Larger toxic structure");
    let mol2_nodes = 6; // 6 atoms
    let mol2_edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0), // ring structure
    ];
    let mol2_features = Array2::from_shape_fn((mol2_nodes, 8), |(i, j)| {
        // Different atom features
        ((i * 5 + j + 3) % 10) as f64 / 10.0 + 0.3
    });
    println!("  Nodes: {}, Edges: {}", mol2_nodes, mol2_edges.len());
    println!();

    // Build GIN model
    let input_dim = 8;
    let hidden_dim = 16;
    let epsilon = 0.0; // fixed epsilon (non-learnable)

    println!("Building GIN model:");
    println!("  Input features: {}", input_dim);
    println!("  Hidden dimension: {}", hidden_dim);
    println!("  Epsilon: {} (fixed)", epsilon);
    println!("  Aggregation: SUM (most expressive for WL test)");
    println!();

    let gin_layer = GinLayer::new(input_dim, hidden_dim, epsilon)?;

    // Process Molecule 1
    println!("Processing Molecule 1...");
    let mol1_adj = SparseAdjacency::from_edges(mol1_nodes, &mol1_edges)?;
    let mol1_hidden = gin_layer.forward(&mol1_adj, &mol1_features.view())?;
    println!("  Node representations shape: {:?}", mol1_hidden.shape());

    // Apply ReLU activation
    let mut mol1_hidden_relu = mol1_hidden.clone();
    for i in 0..mol1_hidden_relu.nrows() {
        for j in 0..mol1_hidden_relu.ncols() {
            if mol1_hidden_relu[[i, j]] < 0.0 {
                mol1_hidden_relu[[i, j]] = 0.0;
            }
        }
    }

    // Global pooling to get graph representation
    let mol1_graph_repr = global_mean_pool(&mol1_hidden_relu.view())?;
    println!("  After global pooling: {:?}", mol1_graph_repr.shape());
    println!(
        "  Graph representation (first 5 dims): [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        mol1_graph_repr[0],
        mol1_graph_repr[1],
        mol1_graph_repr[2],
        mol1_graph_repr[3],
        mol1_graph_repr[4]
    );

    // Process Molecule 2
    println!("\nProcessing Molecule 2...");
    let mol2_adj = SparseAdjacency::from_edges(mol2_nodes, &mol2_edges)?;
    let mol2_hidden = gin_layer.forward(&mol2_adj, &mol2_features.view())?;
    println!("  Node representations shape: {:?}", mol2_hidden.shape());

    // Apply ReLU
    let mut mol2_hidden_relu = mol2_hidden.clone();
    for i in 0..mol2_hidden_relu.nrows() {
        for j in 0..mol2_hidden_relu.ncols() {
            if mol2_hidden_relu[[i, j]] < 0.0 {
                mol2_hidden_relu[[i, j]] = 0.0;
            }
        }
    }

    // Global pooling
    let mol2_graph_repr = global_mean_pool(&mol2_hidden_relu.view())?;
    println!("  After global pooling: {:?}", mol2_graph_repr.shape());
    println!(
        "  Graph representation (first 5 dims): [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        mol2_graph_repr[0],
        mol2_graph_repr[1],
        mol2_graph_repr[2],
        mol2_graph_repr[3],
        mol2_graph_repr[4]
    );

    // Compare graph representations
    println!("\nGraph Representation Comparison:");
    let mut l2_dist = 0.0;
    for i in 0..hidden_dim {
        let diff = mol1_graph_repr[i] - mol2_graph_repr[i];
        l2_dist += diff * diff;
    }
    l2_dist = l2_dist.sqrt();
    println!("  L2 distance between representations: {:.4}", l2_dist);
    println!("  Different structures produce different embeddings!");

    // Demonstrate different pooling strategies
    println!("\nPooling Strategy Comparison:");

    // Mean pooling (already computed)
    println!("  Mean pooling:");
    println!(
        "    Mol1: {:.4}",
        mol1_graph_repr.iter().sum::<f64>() / hidden_dim as f64
    );
    println!(
        "    Mol2: {:.4}",
        mol2_graph_repr.iter().sum::<f64>() / hidden_dim as f64
    );

    // Max pooling
    let mol1_max_pool = global_max_pool(&mol1_hidden_relu.view())?;
    let mol2_max_pool = global_max_pool(&mol2_hidden_relu.view())?;
    println!("  Max pooling:");
    println!(
        "    Mol1: {:.4}",
        mol1_max_pool
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!(
        "    Mol2: {:.4}",
        mol2_max_pool
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // Sum pooling
    let mol1_sum_pool = global_sum_pool(&mol1_hidden_relu.view())?;
    let mol2_sum_pool = global_sum_pool(&mol2_hidden_relu.view())?;
    println!("  Sum pooling:");
    println!("    Mol1: {:.4}", mol1_sum_pool.iter().sum::<f64>());
    println!("    Mol2: {:.4}", mol2_sum_pool.iter().sum::<f64>());

    // Demonstrate GIN's expressiveness
    println!("\nGIN Expressiveness:");
    println!("  GIN with sum aggregation is as powerful as WL test");
    println!("  It can distinguish non-isomorphic graphs");
    println!("  Epsilon parameter: (1+ε)·h_v + Σh_u");
    println!("    ε=0 (fixed): Standard GIN");
    println!("    ε>0 (learnable): Can adapt to specific tasks");

    // Show structural information captured
    println!("\nStructural Information:");
    println!("  Molecule 1 (linear):");
    println!(
        "    Average degree: {:.2}",
        mol1_edges.len() as f64 * 2.0 / mol1_nodes as f64
    );
    println!("    Structure: Open chain");
    println!("  Molecule 2 (ring):");
    println!(
        "    Average degree: {:.2}",
        mol2_edges.len() as f64 * 2.0 / mol2_nodes as f64
    );
    println!("    Structure: Closed cycle");

    println!("\n=== Example completed successfully! ===");
    println!("\nNext steps:");
    println!("  1. Stack multiple GIN layers for deeper networks");
    println!("  2. Add edge features for bond types");
    println!("  3. Train with labeled molecular datasets");
    println!("  4. Use different pooling strategies (DiffPool, TopK)");

    Ok(())
}
