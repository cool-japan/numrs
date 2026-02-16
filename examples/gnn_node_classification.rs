//! Graph Neural Network Node Classification Example
//!
//! This example demonstrates node classification using Graph Convolutional Networks (GCN)
//! on a citation network. We classify papers (nodes) based on their content and citation
//! patterns (edges).
//!
//! # Task
//!
//! Given a graph of academic papers where:
//! - Nodes represent papers
//! - Edges represent citations between papers
//! - Node features represent paper content (bag-of-words)
//! - Goal: Classify papers into research topics
//!
//! # Method
//!
//! We use a 2-layer GCN:
//! 1. Layer 1: Transform features from input_dim to hidden_dim
//! 2. ReLU activation
//! 3. Layer 2: Transform from hidden_dim to num_classes
//! 4. Softmax for class probabilities
//!
//! # Reference
//!
//! Kipf & Welling (2017) - "Semi-Supervised Classification with Graph Convolutional Networks"

use numrs2::new_modules::nn::graph::*;
use scirs2_core::ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GCN Node Classification Example ===\n");

    // Create a small citation network (Zachary's Karate Club-like structure)
    // 7 nodes representing papers in different research areas
    let num_nodes = 7;
    let edges = vec![
        // Area 1: Machine Learning papers (nodes 0, 1, 2)
        (0, 1),
        (1, 0), // bidirectional citations
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
        // Area 2: Computer Vision papers (nodes 3, 4)
        (3, 4),
        (4, 3),
        // Cross-area citations
        (2, 3),
        (3, 2), // ML to CV bridge
        // Area 3: NLP papers (nodes 5, 6)
        (5, 6),
        (6, 5),
        (1, 5),
        (5, 1), // ML to NLP bridge
    ];

    println!("Citation Network:");
    println!("  Nodes: {} papers", num_nodes);
    println!("  Edges: {} citations", edges.len());
    println!("  Research areas: ML (0,1,2), CV (3,4), NLP (5,6)\n");

    // Create adjacency matrix
    let adj = AdjacencyMatrix::<f64>::from_edges(num_nodes, &edges)?;
    println!("Adjacency matrix created");

    // Compute node degrees
    let degrees = adj.degree_matrix()?;
    println!("Node degrees (citation counts):");
    for (i, &deg) in degrees.iter().enumerate() {
        println!("  Paper {}: {} citations", i, deg);
    }
    println!();

    // Create synthetic node features (10-dimensional bag-of-words)
    // In practice, these would be derived from paper abstracts/content
    let node_features = Array2::from_shape_fn((num_nodes, 10), |(i, j)| {
        // Papers in same area have similar features
        let area = if i <= 2 {
            0
        } else if i <= 4 {
            1
        } else {
            2
        };
        let base = (area * 3 + j) as f64 / 10.0;
        let noise = ((i * 7 + j * 13) % 20) as f64 / 100.0;
        base + noise
    });

    println!("Node features shape: {:?}", node_features.shape());
    println!(
        "Feature range: [{:.3}, {:.3}]",
        node_features.iter().cloned().fold(f64::INFINITY, f64::min),
        node_features
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!();

    // Build 2-layer GCN
    let input_dim = 10;
    let hidden_dim = 16;
    let num_classes = 3; // ML, CV, NLP

    println!("Building GCN architecture:");
    println!("  Layer 1: {} -> {} (GCN + ReLU)", input_dim, hidden_dim);
    println!("  Layer 2: {} -> {} (GCN)", hidden_dim, num_classes);
    println!();

    let gcn_layer1 = GcnLayer::new(input_dim, hidden_dim)?;
    let gcn_layer2 = GcnLayer::new(hidden_dim, num_classes)?;

    // Forward pass - Layer 1
    println!("Forward pass...");
    let hidden = gcn_layer1.forward(&adj, &node_features.view())?;
    println!("  After layer 1: shape {:?}", hidden.shape());

    // Apply ReLU activation
    let mut hidden_relu = hidden.clone();
    for i in 0..hidden_relu.nrows() {
        for j in 0..hidden_relu.ncols() {
            if hidden_relu[[i, j]] < 0.0 {
                hidden_relu[[i, j]] = 0.0;
            }
        }
    }
    println!(
        "  After ReLU: {} activations zeroed",
        hidden.iter().filter(|&&x| x < 0.0).count()
    );

    // Forward pass - Layer 2
    let logits = gcn_layer2.forward(&adj, &hidden_relu.view())?;
    println!("  After layer 2: shape {:?}", logits.shape());

    // Apply softmax to get class probabilities
    let mut predictions = Array2::zeros((num_nodes, num_classes));
    for i in 0..num_nodes {
        let mut max_logit = logits[[i, 0]];
        for j in 1..num_classes {
            if logits[[i, j]] > max_logit {
                max_logit = logits[[i, j]];
            }
        }

        let mut sum_exp = 0.0;
        for j in 0..num_classes {
            sum_exp += (logits[[i, j]] - max_logit).exp();
        }

        for j in 0..num_classes {
            predictions[[i, j]] = (logits[[i, j]] - max_logit).exp() / sum_exp;
        }
    }

    println!("\nNode Classification Results:");
    println!("{:-<60}", "");
    println!(
        "{:<8} {:^15} {:^15} {:^15}",
        "Paper", "ML Prob", "CV Prob", "NLP Prob"
    );
    println!("{:-<60}", "");

    for i in 0..num_nodes {
        let area_label = if i <= 2 {
            "ML*"
        } else if i <= 4 {
            "CV*"
        } else {
            "NLP*"
        };
        println!(
            "{:<6} {} {:^15.4} {:^15.4} {:^15.4}",
            i,
            area_label,
            predictions[[i, 0]],
            predictions[[i, 1]],
            predictions[[i, 2]]
        );
    }
    println!("{:-<60}", "");
    println!("* indicates ground truth label");

    // Analyze graph structure impact
    println!("\nGraph Structure Analysis:");
    println!("  Symmetric normalization preserves total probability mass");
    println!("  Node features are aggregated from citation neighborhood");
    println!("  Papers with similar citations get similar representations");

    // Compute prediction accuracy (with random initialization, accuracy will be low)
    let mut correct = 0;
    for i in 0..num_nodes {
        let true_class = if i <= 2 {
            0
        } else if i <= 4 {
            1
        } else {
            2
        };
        let mut pred_class = 0;
        let mut max_prob = predictions[[i, 0]];
        for j in 1..num_classes {
            if predictions[[i, j]] > max_prob {
                max_prob = predictions[[i, j]];
                pred_class = j;
            }
        }
        if pred_class == true_class {
            correct += 1;
        }
    }

    println!("\nPerformance (random initialization):");
    println!(
        "  Accuracy: {}/{} = {:.1}%",
        correct,
        num_nodes,
        100.0 * correct as f64 / num_nodes as f64
    );
    println!("  Note: Train the network to improve accuracy!");

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
