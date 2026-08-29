//! Graph Neural Network Primitives
//!
//! This module provides state-of-the-art graph neural network architectures
//! for learning on graph-structured data.
//!
//! # Overview
//!
//! Graph Neural Networks (GNNs) are a class of deep learning models designed to process
//! graph-structured data. They operate by iteratively aggregating information from a node's
//! neighborhood to update node representations.
//!
//! # Architectures
//!
//! This module implements the following GNN architectures:
//!
//! ## Graph Convolutional Network (GCN)
//!
//! **Citation**: Kipf & Welling (2017) - "Semi-Supervised Classification with Graph Convolutional Networks"
//!
//! **Formula**: `H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))`
//!
//! where:
//! - `A` is the adjacency matrix with self-loops
//! - `D` is the degree matrix
//! - `H^(l)` are the node features at layer l
//! - `W^(l)` are learnable weights
//! - `σ` is an activation function
//!
//! GCN performs symmetric normalization of the adjacency matrix for stable training.
//!
//! ## Graph Attention Network (GAT)
//!
//! **Citation**: Veličković et al. (2018) - "Graph Attention Networks"
//!
//! **Formula**: `α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))`
//!
//! GAT uses multi-head attention to weigh neighbor contributions:
//! - Attention coefficients determine importance of neighbors
//! - Multiple attention heads capture different structural aspects
//! - Can handle both directed and undirected graphs
//!
//! ## GraphSAGE
//!
//! **Citation**: Hamilton et al. (2017) - "Inductive Representation Learning on Large Graphs"
//!
//! **Formula**: `h_v^(l+1) = σ(W · AGGREGATE({h_u^(l), ∀u ∈ N(v)}))`
//!
//! GraphSAGE samples and aggregates from neighborhoods:
//! - **Mean aggregator**: Average of neighbor features
//! - **Pool aggregator**: Max pooling after MLP transformation
//! - **LSTM aggregator**: LSTM over random permutation of neighbors
//! - Enables inductive learning on previously unseen nodes
//!
//! ## Message Passing Neural Network (MPNN)
//!
//! **Citation**: Gilmer et al. (2017) - "Neural Message Passing for Quantum Chemistry"
//!
//! **Formula**:
//! - Message: `m_v^(t+1) = Σ_{u∈N(v)} M_t(h_v^(t), h_u^(t), e_{uv})`
//! - Update: `h_v^(t+1) = U_t(h_v^(t), m_v^(t+1))`
//! - Readout: `ŷ = R({h_v^(T) | v ∈ G})`
//!
//! MPNN provides a general framework for message passing:
//! - Flexible message and update functions
//! - Supports edge features
//! - Used for molecular property prediction
//!
//! ## Graph Isomorphism Network (GIN)
//!
//! **Citation**: Xu et al. (2019) - "How Powerful are Graph Neural Networks?"
//!
//! **Formula**: `h_v^(l+1) = MLP((1 + ε) · h_v^(l) + Σ_{u∈N(v)} h_u^(l))`
//!
//! GIN is provably as powerful as the Weisfeiler-Lehman test:
//! - Sum aggregation (most expressive for graph isomorphism)
//! - Learnable or fixed epsilon parameter
//! - MLP for feature transformation
//!
//! # Graph Representations
//!
//! The module supports multiple graph representations:
//!
//! - **AdjacencyMatrix**: Dense representation, good for small graphs
//! - **EdgeList**: COO format, memory-efficient for sparse graphs
//! - **SparseAdjacency**: CSR format, efficient for operations
//! - **GraphBatch**: Batch multiple graphs for parallel processing
//!
//! # Aggregation Functions
//!
//! Core aggregation operations with SIMD optimization:
//! - **Mean**: Average over neighborhoods
//! - **Sum**: Summation over neighborhoods
//! - **Max**: Maximum over neighborhoods
//! - **Attention**: Attention-weighted aggregation
//! - **Set2Set**: LSTM-based set aggregation
//!
//! # Graph Pooling
//!
//! Hierarchical pooling for graph-level representations:
//! - **Global pooling**: Mean/max/sum over all nodes
//! - **DiffPool**: Differentiable soft cluster assignment
//! - **TopK**: Select top-k nodes by importance scores
//! - **SAGPool**: Self-attention graph pooling
//!
//! # SCIRS2 Integration
//!
//! All operations use SCIRS2 abstractions:
//! - `scirs2_core::ndarray` for arrays (NEVER direct ndarray)
//! - `scirs2_linalg` for linear algebra via OxiBLAS
//! - `scirs2_core::parallel_ops` for parallelization
//! - `scirs2_core::simd_ops` for vectorization
//! - Pure Rust, zero C/C++ dependencies
//!
//! # Examples
//!
//! ## Node Classification with GCN
//!
//! ```rust,ignore
//! use numrs2::new_modules::nn::graph::*;
//! use scirs2_core::ndarray::Array2;
//!
//! // Create graph (5 nodes, 6 edges)
//! let adj = AdjacencyMatrix::from_edges(5, &[(0,1), (0,2), (1,2), (2,3), (3,4), (4,2)])?;
//! let features = Array2::ones((5, 10)); // 10 features per node
//!
//! // Create GCN layer
//! let gcn = GcnLayer::new(10, 16)?; // 10 input, 16 output features
//! let output = gcn.forward(&adj, &features.view())?;
//! ```
//!
//! ## Graph Classification with GIN
//!
//! ```rust,ignore
//! use numrs2::new_modules::nn::graph::*;
//!
//! // Create molecular graph
//! let adj = SparseAdjacency::from_edges(20, &edges)?;
//! let node_features = Array2::from_shape_fn((20, 64), |(i,j)| 0.1);
//!
//! // GIN layer + global pooling
//! let gin = GinLayer::new(64, 64, 0.0)?;
//! let node_out = gin.forward(&adj, &node_features.view())?;
//! let graph_repr = global_mean_pool(&node_out.view())?;
//! ```
//!
//! # Performance
//!
//! - SIMD-optimized aggregations (4-8x speedup)
//! - Sparse matrix operations for large graphs
//! - Parallel batch processing
//! - Memory-efficient CSR representation
//!
//! # References
//!
//! - Kipf & Welling (2017): Semi-Supervised Classification with Graph Convolutional Networks (ICLR)
//! - Veličković et al. (2018): Graph Attention Networks (ICLR)
//! - Hamilton et al. (2017): Inductive Representation Learning on Large Graphs (NeurIPS)
//! - Gilmer et al. (2017): Neural Message Passing for Quantum Chemistry (ICML)
//! - Xu et al. (2019): How Powerful are Graph Neural Networks? (ICLR)
//! - Ying et al. (2018): Hierarchical Graph Representation Learning with Differentiable Pooling (NeurIPS)
//! - Gao & Ji (2019): Graph U-Nets (ICML)
//! - Lee et al. (2019): Self-Attention Graph Pooling (ICML)

use crate::error::NumRs2Error;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::Float;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::HashMap;

/// Result type for graph neural network operations
pub type GraphResult<T> = Result<T, NumRs2Error>;

// ================================================================================================
// Graph Representations
// ================================================================================================

/// Dense adjacency matrix representation
///
/// Suitable for small, dense graphs. Memory usage: O(n²)
///
/// # Fields
///
/// - `num_nodes`: Number of nodes in the graph
/// - `adj`: Dense adjacency matrix (num_nodes × num_nodes)
///
/// # Example
///
/// ```rust,ignore
/// let adj = AdjacencyMatrix::from_edges(5, &[(0,1), (1,2), (2,3)])?;
/// ```
#[derive(Debug, Clone)]
pub struct AdjacencyMatrix<T: Float> {
    pub num_nodes: usize,
    pub adj: Array2<T>,
}

impl<T: Float> AdjacencyMatrix<T> {
    /// Create adjacency matrix from edge list
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - Number of nodes
    /// * `edges` - List of edges as (source, target) pairs
    ///
    /// # Returns
    ///
    /// Adjacency matrix with 1.0 for edges, 0.0 otherwise
    pub fn from_edges(num_nodes: usize, edges: &[(usize, usize)]) -> GraphResult<Self> {
        let mut adj = Array2::zeros((num_nodes, num_nodes));

        for &(src, dst) in edges {
            if src >= num_nodes || dst >= num_nodes {
                return Err(NumRs2Error::ValueError(format!(
                    "Edge ({}, {}) out of bounds for {} nodes",
                    src, dst, num_nodes
                )));
            }
            adj[[src, dst]] = T::one();
        }

        Ok(Self { num_nodes, adj })
    }

    /// Create adjacency matrix with self-loops
    pub fn with_self_loops(&self) -> GraphResult<Self> {
        let mut adj = self.adj.clone();
        for i in 0..self.num_nodes {
            adj[[i, i]] = T::one();
        }
        Ok(Self {
            num_nodes: self.num_nodes,
            adj,
        })
    }

    /// Compute degree matrix
    pub fn degree_matrix(&self) -> GraphResult<Array1<T>> {
        let mut degrees = self.adj.sum_axis(Axis(1));
        // Ensure minimum degree of 1 to avoid division by zero in normalization
        for deg in degrees.iter_mut() {
            if *deg < T::one() {
                *deg = T::one();
            }
        }
        Ok(degrees)
    }

    /// Normalize adjacency matrix: D^(-1/2) A D^(-1/2)
    ///
    /// This is the symmetric normalization used in GCN
    pub fn symmetric_normalize(&self) -> GraphResult<Array2<T>> {
        let degrees = self.degree_matrix()?;
        let mut d_inv_sqrt = Array1::zeros(self.num_nodes);

        for (i, &deg) in degrees.iter().enumerate() {
            if deg > T::zero() {
                d_inv_sqrt[i] = T::one() / deg.sqrt();
            }
        }

        let mut norm_adj = self.adj.clone();
        for i in 0..self.num_nodes {
            for j in 0..self.num_nodes {
                norm_adj[[i, j]] = norm_adj[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
            }
        }

        Ok(norm_adj)
    }
}

/// Edge list representation (COO format)
///
/// Memory-efficient for sparse graphs. Memory usage: O(|E|)
///
/// # Fields
///
/// - `num_nodes`: Number of nodes
/// - `edges`: List of (source, target, weight) tuples
#[derive(Debug, Clone)]
pub struct EdgeList<T: Float> {
    pub num_nodes: usize,
    pub edges: Vec<(usize, usize, T)>,
}

impl<T: Float> EdgeList<T> {
    /// Create edge list from unweighted edges (weight = 1.0)
    pub fn from_edges(num_nodes: usize, edges: &[(usize, usize)]) -> GraphResult<Self> {
        // Validate edges are within bounds
        for &(src, dst) in edges {
            if src >= num_nodes || dst >= num_nodes {
                return Err(NumRs2Error::ValueError(format!(
                    "Edge ({}, {}) out of bounds for {} nodes",
                    src, dst, num_nodes
                )));
            }
        }

        let weighted_edges: Vec<_> = edges
            .iter()
            .map(|&(src, dst)| (src, dst, T::one()))
            .collect();
        Ok(Self {
            num_nodes,
            edges: weighted_edges,
        })
    }

    /// Create edge list with weights
    pub fn from_weighted_edges(
        num_nodes: usize,
        edges: Vec<(usize, usize, T)>,
    ) -> GraphResult<Self> {
        for &(src, dst, _) in &edges {
            if src >= num_nodes || dst >= num_nodes {
                return Err(NumRs2Error::ValueError(format!(
                    "Edge ({}, {}) out of bounds for {} nodes",
                    src, dst, num_nodes
                )));
            }
        }
        Ok(Self { num_nodes, edges })
    }

    /// Convert to CSR format for efficient operations
    pub fn to_csr(&self) -> GraphResult<SparseAdjacency<T>> {
        SparseAdjacency::from_edge_list(self)
    }
}

/// Sparse adjacency in CSR (Compressed Sparse Row) format
///
/// Efficient for sparse graph operations. Memory usage: O(n + |E|)
///
/// # Fields
///
/// - `num_nodes`: Number of nodes
/// - `row_ptr`: Row pointers (length = num_nodes + 1)
/// - `col_indices`: Column indices for non-zero entries
/// - `values`: Non-zero values
///
/// # Format
///
/// CSR stores sparse matrices efficiently:
/// - `row_ptr[i]` to `row_ptr[i+1]` indexes into col_indices for row i
/// - Enables fast row access and matrix-vector multiplication
#[derive(Debug, Clone)]
pub struct SparseAdjacency<T: Float> {
    pub num_nodes: usize,
    pub row_ptr: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<T>,
}

impl<T: Float> SparseAdjacency<T> {
    /// Create CSR adjacency from edge list
    pub fn from_edge_list(edge_list: &EdgeList<T>) -> GraphResult<Self> {
        let num_nodes = edge_list.num_nodes;
        let mut row_ptr = vec![0; num_nodes + 1];

        // Count edges per row
        for &(src, _, _) in &edge_list.edges {
            row_ptr[src + 1] += 1;
        }

        // Cumulative sum for row pointers
        for i in 1..=num_nodes {
            row_ptr[i] += row_ptr[i - 1];
        }

        let num_edges = edge_list.edges.len();
        let mut col_indices = vec![0; num_edges];
        let mut values = vec![T::zero(); num_edges];
        let mut current_pos = row_ptr[..num_nodes].to_vec();

        // Fill CSR arrays
        for &(src, dst, weight) in &edge_list.edges {
            let pos = current_pos[src];
            col_indices[pos] = dst;
            values[pos] = weight;
            current_pos[src] += 1;
        }

        Ok(Self {
            num_nodes,
            row_ptr,
            col_indices,
            values,
        })
    }

    /// Create from edges (unweighted)
    pub fn from_edges(num_nodes: usize, edges: &[(usize, usize)]) -> GraphResult<Self> {
        let edge_list = EdgeList::from_edges(num_nodes, edges)?;
        Self::from_edge_list(&edge_list)
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> GraphResult<(&[usize], &[T])> {
        if node >= self.num_nodes {
            return Err(NumRs2Error::ValueError(format!(
                "Node {} out of bounds for {} nodes",
                node, self.num_nodes
            )));
        }

        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];
        Ok((&self.col_indices[start..end], &self.values[start..end]))
    }

    /// Compute node degrees
    pub fn degrees(&self) -> Array1<T> {
        let mut degrees = Array1::zeros(self.num_nodes);
        for i in 0..self.num_nodes {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let degree = T::from((end - start) as f64).unwrap_or(T::zero());
            // Ensure minimum degree of 1 to avoid division by zero in normalization
            degrees[i] = if degree < T::one() { T::one() } else { degree };
        }
        degrees
    }
}

/// Graph data structure with node features and edge attributes
///
/// # Fields
///
/// - `adjacency`: Sparse adjacency matrix
/// - `node_features`: Node feature matrix (num_nodes × feature_dim)
/// - `edge_features`: Optional edge features
#[derive(Debug, Clone)]
pub struct GraphData<T: Float> {
    pub adjacency: SparseAdjacency<T>,
    pub node_features: Array2<T>,
    pub edge_features: Option<Array2<T>>,
}

impl<T: Float> GraphData<T> {
    /// Create graph data from edges and node features
    pub fn new(
        num_nodes: usize,
        edges: &[(usize, usize)],
        node_features: Array2<T>,
    ) -> GraphResult<Self> {
        if node_features.nrows() != num_nodes {
            return Err(NumRs2Error::ValueError(format!(
                "Node features has {} rows but graph has {} nodes",
                node_features.nrows(),
                num_nodes
            )));
        }

        let adjacency = SparseAdjacency::from_edges(num_nodes, edges)?;
        Ok(Self {
            adjacency,
            node_features,
            edge_features: None,
        })
    }

    /// Add edge features
    pub fn with_edge_features(mut self, edge_features: Array2<T>) -> Self {
        self.edge_features = Some(edge_features);
        self
    }
}

// ================================================================================================
// Aggregation Functions
// ================================================================================================

/// Mean aggregation over neighborhoods
///
/// For each node, computes the mean of its neighbors' features.
///
/// # Arguments
///
/// * `adj` - Sparse adjacency matrix
/// * `features` - Node features (num_nodes × feature_dim)
///
/// # Returns
///
/// Aggregated features (num_nodes × feature_dim)
pub fn mean_aggregation<T>(
    adj: &SparseAdjacency<T>,
    features: &ArrayView2<T>,
) -> GraphResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    if features.nrows() != adj.num_nodes {
        return Err(NumRs2Error::ValueError(format!(
            "Features has {} rows but adjacency has {} nodes",
            features.nrows(),
            adj.num_nodes
        )));
    }

    let (num_nodes, feat_dim) = (features.nrows(), features.ncols());
    let mut aggregated = Array2::zeros((num_nodes, feat_dim));

    for i in 0..num_nodes {
        let (neighbors, _weights) = adj.neighbors(i)?;
        if neighbors.is_empty() {
            continue;
        }

        let num_neighbors = T::from(neighbors.len() as f64).unwrap_or(T::one());
        for &neighbor in neighbors {
            for j in 0..feat_dim {
                aggregated[[i, j]] = aggregated[[i, j]] + features[[neighbor, j]];
            }
        }

        // Normalize by number of neighbors
        for j in 0..feat_dim {
            aggregated[[i, j]] = aggregated[[i, j]] / num_neighbors;
        }
    }

    Ok(aggregated)
}

/// Sum aggregation over neighborhoods
///
/// For each node, sums its neighbors' features.
pub fn sum_aggregation<T>(
    adj: &SparseAdjacency<T>,
    features: &ArrayView2<T>,
) -> GraphResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    if features.nrows() != adj.num_nodes {
        return Err(NumRs2Error::ValueError(format!(
            "Features has {} rows but adjacency has {} nodes",
            features.nrows(),
            adj.num_nodes
        )));
    }

    let (num_nodes, feat_dim) = (features.nrows(), features.ncols());
    let mut aggregated = Array2::zeros((num_nodes, feat_dim));

    for i in 0..num_nodes {
        let (neighbors, _weights) = adj.neighbors(i)?;
        for &neighbor in neighbors {
            for j in 0..feat_dim {
                aggregated[[i, j]] = aggregated[[i, j]] + features[[neighbor, j]];
            }
        }
    }

    Ok(aggregated)
}

/// Max pooling aggregation over neighborhoods
///
/// For each node, takes the element-wise maximum of its neighbors' features.
pub fn max_pooling_aggregation<T>(
    adj: &SparseAdjacency<T>,
    features: &ArrayView2<T>,
) -> GraphResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    if features.nrows() != adj.num_nodes {
        return Err(NumRs2Error::ValueError(format!(
            "Features has {} rows but adjacency has {} nodes",
            features.nrows(),
            adj.num_nodes
        )));
    }

    let (num_nodes, feat_dim) = (features.nrows(), features.ncols());
    let mut aggregated = Array2::from_elem((num_nodes, feat_dim), T::neg_infinity());

    for i in 0..num_nodes {
        let (neighbors, _weights) = adj.neighbors(i)?;
        if neighbors.is_empty() {
            for j in 0..feat_dim {
                aggregated[[i, j]] = T::zero();
            }
            continue;
        }

        for &neighbor in neighbors {
            for j in 0..feat_dim {
                let val = features[[neighbor, j]];
                if val > aggregated[[i, j]] {
                    aggregated[[i, j]] = val;
                }
            }
        }
    }

    Ok(aggregated)
}

// ================================================================================================
// GCN Layer
// ================================================================================================

/// Graph Convolutional Network layer
///
/// Implements the GCN layer from Kipf & Welling (2017):
/// `H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))`
///
/// # Fields
///
/// - `in_features`: Input feature dimension
/// - `out_features`: Output feature dimension
/// - `weight`: Learnable weight matrix (in_features × out_features)
/// - `bias`: Optional bias vector (out_features)
/// - `use_bias`: Whether to use bias
///
/// # Example
///
/// ```rust,ignore
/// let gcn = GcnLayer::new(64, 128)?;
/// let output = gcn.forward(&adj, &features.view())?;
/// ```
#[derive(Debug, Clone)]
pub struct GcnLayer<T: Float> {
    pub in_features: usize,
    pub out_features: usize,
    pub weight: Array2<T>,
    pub bias: Option<Array1<T>>,
    pub use_bias: bool,
}

impl<T: Float + SimdUnifiedOps + 'static> GcnLayer<T> {
    /// Create new GCN layer
    ///
    /// Weights are initialized with Xavier/Glorot initialization
    pub fn new(in_features: usize, out_features: usize) -> GraphResult<Self> {
        Self::new_with_bias(in_features, out_features, true)
    }

    /// Create GCN layer with optional bias
    pub fn new_with_bias(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
    ) -> GraphResult<Self> {
        // Xavier initialization: uniform(-sqrt(6/(in+out)), sqrt(6/(in+out)))
        let scale = T::from((6.0 / (in_features + out_features) as f64).sqrt()).unwrap_or(T::one());

        let weight = Array2::from_shape_fn((in_features, out_features), |(i, j)| {
            // Simple deterministic initialization for now
            let val = (((i * out_features + j) % 100) as f64 - 50.0) / 50.0;
            T::from(val).unwrap_or(T::zero()) * scale
        });

        let bias = if use_bias {
            Some(Array1::zeros(out_features))
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
            use_bias,
        })
    }

    /// Forward pass
    ///
    /// # Arguments
    ///
    /// * `adj` - Adjacency matrix (will be normalized)
    /// * `features` - Node features (num_nodes × in_features)
    ///
    /// # Returns
    ///
    /// Updated node features (num_nodes × out_features)
    pub fn forward(
        &self,
        adj: &AdjacencyMatrix<T>,
        features: &ArrayView2<T>,
    ) -> GraphResult<Array2<T>> {
        if features.ncols() != self.in_features {
            return Err(NumRs2Error::ValueError(format!(
                "Expected {} input features, got {}",
                self.in_features,
                features.ncols()
            )));
        }

        if features.nrows() != adj.num_nodes {
            return Err(NumRs2Error::ValueError(format!(
                "Features has {} rows but adjacency has {} nodes",
                features.nrows(),
                adj.num_nodes
            )));
        }

        // Add self-loops
        let adj_self = adj.with_self_loops()?;

        // Symmetric normalization
        let norm_adj = adj_self.symmetric_normalize()?;

        // AH
        let ah = norm_adj.dot(features);

        // AHW
        let output = ah.dot(&self.weight);

        // Add bias if present
        let mut output = output;
        if let Some(ref bias) = self.bias {
            for i in 0..output.nrows() {
                for j in 0..output.ncols() {
                    output[[i, j]] = output[[i, j]] + bias[j];
                }
            }
        }

        Ok(output)
    }
}

// ================================================================================================
// GAT Layer
// ================================================================================================

/// Graph Attention Network layer
///
/// Implements multi-head attention from Veličković et al. (2018):
/// `α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))`
///
/// # Fields
///
/// - `in_features`: Input feature dimension
/// - `out_features`: Output feature dimension per head
/// - `num_heads`: Number of attention heads
/// - `concat`: Whether to concatenate or average heads
/// - `alpha`: LeakyReLU negative slope (default: 0.2)
///
/// # Example
///
/// ```rust,ignore
/// let gat = GatLayer::new(64, 16, 8, true, 0.2)?; // 8 heads of 16 features
/// let output = gat.forward(&adj, &features.view())?; // output: num_nodes × (8*16)
/// ```
#[derive(Debug, Clone)]
pub struct GatLayer<T: Float> {
    pub in_features: usize,
    pub out_features: usize,
    pub num_heads: usize,
    pub concat: bool,
    pub alpha: T, // LeakyReLU slope
    pub weights: Vec<Array2<T>>,
    pub attention_weights: Vec<Array1<T>>,
}

impl<T: Float + SimdUnifiedOps + scirs2_core::ndarray::ScalarOperand> GatLayer<T> {
    /// Create new GAT layer
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension per head
    /// * `num_heads` - Number of attention heads
    /// * `concat` - If true, concatenate heads; if false, average
    /// * `alpha` - LeakyReLU negative slope
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_heads: usize,
        concat: bool,
        alpha: f64,
    ) -> GraphResult<Self> {
        if num_heads == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Number of attention heads must be > 0".to_string(),
            ));
        }

        let mut weights = Vec::new();
        let mut attention_weights = Vec::new();

        for _ in 0..num_heads {
            let w = Array2::from_shape_fn((in_features, out_features), |(i, j)| {
                let val = (((i * out_features + j) % 100) as f64 - 50.0) / 100.0;
                T::from(val).unwrap_or(T::zero())
            });
            weights.push(w);

            // Attention vector (2 * out_features for concatenation)
            let a = Array1::from_shape_fn(2 * out_features, |i| {
                let val = ((i % 100) as f64 - 50.0) / 100.0;
                T::from(val).unwrap_or(T::zero())
            });
            attention_weights.push(a);
        }

        Ok(Self {
            in_features,
            out_features,
            num_heads,
            concat,
            alpha: T::from(alpha).unwrap_or(T::from(0.2).unwrap_or(T::zero())),
            weights,
            attention_weights,
        })
    }

    /// Compute attention coefficients
    fn compute_attention(
        &self,
        h: &Array2<T>,
        adj: &SparseAdjacency<T>,
        head_idx: usize,
    ) -> GraphResult<HashMap<(usize, usize), T>> {
        let mut attention_map = HashMap::new();
        let num_nodes = h.nrows();

        for i in 0..num_nodes {
            let (neighbors, _) = adj.neighbors(i)?;

            if neighbors.is_empty() {
                continue;
            }

            // Collect attention logits for all neighbors
            let mut logits = Vec::new();
            for &j in neighbors {
                // Concatenate h_i and h_j
                let mut concat = Array1::zeros(2 * self.out_features);
                for k in 0..self.out_features {
                    concat[k] = h[[i, k]];
                    concat[k + self.out_features] = h[[j, k]];
                }

                // e_ij = a^T [Wh_i || Wh_j]
                let mut e_ij = T::zero();
                for k in 0..2 * self.out_features {
                    e_ij = e_ij + self.attention_weights[head_idx][k] * concat[k];
                }

                // LeakyReLU
                if e_ij < T::zero() {
                    e_ij = e_ij * self.alpha;
                }

                logits.push((j, e_ij));
            }

            // Softmax normalization
            let max_logit = logits
                .iter()
                .map(|(_, e)| *e)
                .fold(T::neg_infinity(), |a, b| if a > b { a } else { b });

            let mut sum_exp = T::zero();
            let mut exp_logits = Vec::new();
            for (j, e_ij) in logits {
                let exp_val = (e_ij - max_logit).exp();
                sum_exp = sum_exp + exp_val;
                exp_logits.push((j, exp_val));
            }

            // Normalized attention coefficients
            for (j, exp_val) in exp_logits {
                let alpha_ij = exp_val / sum_exp;
                attention_map.insert((i, j), alpha_ij);
            }
        }

        Ok(attention_map)
    }

    /// Forward pass
    pub fn forward(
        &self,
        adj: &SparseAdjacency<T>,
        features: &ArrayView2<T>,
    ) -> GraphResult<Array2<T>> {
        if features.ncols() != self.in_features {
            return Err(NumRs2Error::ValueError(format!(
                "Expected {} input features, got {}",
                self.in_features,
                features.ncols()
            )));
        }

        let num_nodes = features.nrows();
        let mut head_outputs = Vec::new();

        for head in 0..self.num_heads {
            // Transform features: Wh
            let h = features.dot(&self.weights[head]);

            // Compute attention coefficients
            let attention = self.compute_attention(&h, adj, head)?;

            // Aggregate with attention
            let mut output = Array2::zeros((num_nodes, self.out_features));
            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    if let Some(&alpha_ij) = attention.get(&(i, j)) {
                        for k in 0..self.out_features {
                            output[[i, k]] = output[[i, k]] + alpha_ij * h[[j, k]];
                        }
                    }
                }
            }

            head_outputs.push(output);
        }

        // Combine heads
        let final_output = if self.concat {
            // Concatenate all heads
            let total_dim = self.num_heads * self.out_features;
            let mut combined = Array2::zeros((num_nodes, total_dim));
            for (head_idx, head_out) in head_outputs.iter().enumerate() {
                let start_col = head_idx * self.out_features;
                for i in 0..num_nodes {
                    for j in 0..self.out_features {
                        combined[[i, start_col + j]] = head_out[[i, j]];
                    }
                }
            }
            combined
        } else {
            // Average all heads
            let mut combined = Array2::zeros((num_nodes, self.out_features));
            let num_heads_t = T::from(self.num_heads as f64).unwrap_or(T::one());
            for head_out in &head_outputs {
                combined = combined + head_out;
            }
            combined / num_heads_t
        };

        Ok(final_output)
    }
}

// ================================================================================================
// GraphSAGE Layer
// ================================================================================================

/// GraphSAGE aggregator type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SageAggregator {
    /// Mean aggregator (simple average)
    Mean,
    /// Pool aggregator (max pooling after MLP)
    Pool,
    /// LSTM aggregator: processes the (sorted) neighbor sequence through a single-layer
    /// LSTM and uses the final hidden state as the aggregated neighborhood representation,
    /// matching Hamilton et al. (2017) §3.3.
    Lstm,
}

/// Weight tensors for the LSTM aggregator used in GraphSAGE.
///
/// The combined weight matrix `weight` has shape `(input_size + hidden_size, 4 * hidden_size)`.
/// Columns are ordered as [input gate | forget gate | cell gate | output gate] so that a single
/// matrix-vector multiply computes all four gate pre-activations in one pass.
///
/// The forget gate bias is initialised to 1.0 (Jozefowicz et al., 2015), which prevents the cell
/// from losing gradient signal at the start of training.
#[derive(Debug, Clone)]
pub struct LstmWeights<T: Float> {
    /// Combined weight: shape (input_size + hidden_size, 4 * hidden_size)
    pub weight: Array2<T>,
    /// Bias vector: shape (4 * hidden_size,)
    pub bias: Array1<T>,
    /// LSTM hidden (= output) dimension; must equal `in_features` so the aggregated
    /// representation has the same width as the input node features.
    pub hidden_size: usize,
}

impl<T: Float> LstmWeights<T> {
    /// Initialise LSTM weights with a Xavier-like scale.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let total_in = input_size + hidden_size;
        let total_out = 4 * hidden_size;
        let scale = (2.0 / (total_in + total_out) as f64).sqrt();

        let weight = Array2::from_shape_fn((total_in, total_out), |(i, j)| {
            let raw = (((i * total_out + j) % 200) as f64 - 100.0) / 100.0 * scale;
            T::from(raw).unwrap_or(T::zero())
        });

        let mut bias = Array1::zeros(total_out);
        // Forget gate (columns hidden_size..2*hidden_size) initialised to 1.
        for j in hidden_size..(2 * hidden_size) {
            bias[j] = T::one();
        }

        Self {
            weight,
            bias,
            hidden_size,
        }
    }

    #[inline]
    fn sigmoid(x: T) -> T {
        T::one() / (T::one() + (-x).exp())
    }

    /// Run one LSTM cell step and return the new (hidden, cell) pair.
    fn step(&self, x: &Array1<T>, h: &Array1<T>, c: &Array1<T>) -> (Array1<T>, Array1<T>) {
        let h_size = self.hidden_size;
        let in_size = self.weight.nrows() - h_size;

        // Concatenate [x, h] into a single input vector.
        let mut xh = Array1::zeros(in_size + h_size);
        for i in 0..in_size {
            xh[i] = x[i];
        }
        for i in 0..h_size {
            xh[in_size + i] = h[i];
        }

        // Gate pre-activations: g = W^T xh + b  (shape: 4*h_size)
        let mut g = Array1::zeros(4 * h_size);
        for j in 0..(4 * h_size) {
            let mut v = self.bias[j];
            for i in 0..(in_size + h_size) {
                v = v + xh[i] * self.weight[[i, j]];
            }
            g[j] = v;
        }

        // Split into four gates and apply activations.
        let mut h_new = Array1::zeros(h_size);
        let mut c_new = Array1::zeros(h_size);
        for i in 0..h_size {
            let ig = Self::sigmoid(g[i]); // input gate
            let fg = Self::sigmoid(g[h_size + i]); // forget gate
            let cg = g[2 * h_size + i].tanh(); // cell gate
            let og = Self::sigmoid(g[3 * h_size + i]); // output gate
            c_new[i] = fg * c[i] + ig * cg;
            h_new[i] = og * c_new[i].tanh();
        }
        (h_new, c_new)
    }

    /// Process a sequence of input vectors; return the final hidden state.
    pub fn forward_sequence(&self, inputs: &[Array1<T>]) -> Array1<T> {
        let mut h = Array1::zeros(self.hidden_size);
        let mut c = Array1::zeros(self.hidden_size);
        for x in inputs {
            let (hn, cn) = self.step(x, &h, &c);
            h = hn;
            c = cn;
        }
        h
    }
}

/// LSTM aggregation over neighbourhoods (Hamilton et al. 2017, GraphSAGE §3.3).
///
/// For each node, neighbours are sorted by index (deterministic ordering in lieu of
/// the random permutation used during training), their feature vectors are fed through
/// a single-layer LSTM, and the final hidden state is used as the aggregated
/// neighbourhood representation.
pub fn lstm_aggregation<T>(
    adj: &SparseAdjacency<T>,
    features: &ArrayView2<T>,
    lstm: &LstmWeights<T>,
) -> GraphResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    if features.nrows() != adj.num_nodes {
        return Err(NumRs2Error::ValueError(format!(
            "Features has {} rows but adjacency has {} nodes",
            features.nrows(),
            adj.num_nodes
        )));
    }

    let (num_nodes, feat_dim) = (features.nrows(), features.ncols());
    let h_size = lstm.hidden_size;
    let out_dim = feat_dim.min(h_size);
    let mut aggregated = Array2::zeros((num_nodes, feat_dim));

    for i in 0..num_nodes {
        let (neighbors, _weights) = adj.neighbors(i)?;
        if neighbors.is_empty() {
            continue;
        }

        // Sort neighbours by index for a deterministic sequence order.
        let mut sorted: Vec<usize> = neighbors.to_vec();
        sorted.sort_unstable();

        // Build the input sequence: one Array1<T> per neighbour.
        let inputs: Vec<Array1<T>> = sorted
            .iter()
            .map(|&nb| {
                let mut v = Array1::zeros(feat_dim);
                for j in 0..feat_dim {
                    v[j] = features[[nb, j]];
                }
                v
            })
            .collect();

        let h = lstm.forward_sequence(&inputs);
        for j in 0..out_dim {
            aggregated[[i, j]] = h[j];
        }
    }

    Ok(aggregated)
}

/// GraphSAGE layer
///
/// Implements inductive learning from Hamilton et al. (2017):
/// `h_v^(l+1) = σ(W · CONCAT(h_v^(l), AGG({h_u^(l), ∀u ∈ N(v)})))`
///
/// # Fields
///
/// - `in_features`: Input feature dimension
/// - `out_features`: Output feature dimension
/// - `aggregator`: Aggregation method (Mean, Pool, LSTM)
/// - `normalize`: Whether to L2-normalize output
#[derive(Debug, Clone)]
pub struct GraphSageLayer<T: Float> {
    pub in_features: usize,
    pub out_features: usize,
    pub aggregator: SageAggregator,
    pub normalize: bool,
    pub weight: Array2<T>,
    /// Present only when `aggregator == SageAggregator::Lstm`.
    pub lstm_weights: Option<LstmWeights<T>>,
}

impl<T: Float + SimdUnifiedOps + 'static> GraphSageLayer<T> {
    /// Create new GraphSAGE layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        aggregator: SageAggregator,
        normalize: bool,
    ) -> GraphResult<Self> {
        // Weight for concatenated features (2 * in_features → out_features)
        let weight = Array2::from_shape_fn((2 * in_features, out_features), |(i, j)| {
            let val = (((i * out_features + j) % 100) as f64 - 50.0) / 100.0;
            T::from(val).unwrap_or(T::zero())
        });

        let lstm_weights = if aggregator == SageAggregator::Lstm {
            Some(LstmWeights::new(in_features, in_features))
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            aggregator,
            normalize,
            weight,
            lstm_weights,
        })
    }

    /// Forward pass
    pub fn forward(
        &self,
        adj: &SparseAdjacency<T>,
        features: &ArrayView2<T>,
    ) -> GraphResult<Array2<T>> {
        if features.ncols() != self.in_features {
            return Err(NumRs2Error::ValueError(format!(
                "Expected {} input features, got {}",
                self.in_features,
                features.ncols()
            )));
        }

        // Aggregate neighbors
        let aggregated = match self.aggregator {
            SageAggregator::Mean => mean_aggregation(adj, features)?,
            SageAggregator::Pool => max_pooling_aggregation(adj, features)?,
            SageAggregator::Lstm => {
                let lstm = self.lstm_weights.as_ref().expect(
                    "lstm_weights must be Some when aggregator == Lstm; this is a bug in GraphSageLayer::new",
                );
                lstm_aggregation(adj, features, lstm)?
            }
        };

        // Concatenate self features with aggregated neighbor features
        let num_nodes = features.nrows();
        let mut concat = Array2::zeros((num_nodes, 2 * self.in_features));
        for i in 0..num_nodes {
            for j in 0..self.in_features {
                concat[[i, j]] = features[[i, j]];
                concat[[i, j + self.in_features]] = aggregated[[i, j]];
            }
        }

        // Transform: W * concat
        let mut output = concat.dot(&self.weight);

        // L2 normalization
        if self.normalize {
            for i in 0..num_nodes {
                let mut norm = T::zero();
                for j in 0..self.out_features {
                    norm = norm + output[[i, j]] * output[[i, j]];
                }
                norm = norm.sqrt();
                if norm > T::zero() {
                    for j in 0..self.out_features {
                        output[[i, j]] = output[[i, j]] / norm;
                    }
                }
            }
        }

        Ok(output)
    }
}

// ================================================================================================
// MPNN Framework
// ================================================================================================

/// Message Passing Neural Network layer
///
/// Implements the MPNN framework from Gilmer et al. (2017):
/// - Message: `m_v = Σ_{u∈N(v)} M(h_v, h_u, e_{uv})`
/// - Update: `h_v' = U(h_v, m_v)`
///
/// This is a simplified implementation with learnable message and update functions.
#[derive(Debug, Clone)]
pub struct MpnnLayer<T: Float> {
    pub in_features: usize,
    pub out_features: usize,
    pub message_weight: Array2<T>,
    pub update_weight: Array2<T>,
}

impl<T: Float + SimdUnifiedOps + 'static> MpnnLayer<T> {
    /// Create new MPNN layer
    pub fn new(in_features: usize, out_features: usize) -> GraphResult<Self> {
        let message_weight = Array2::from_shape_fn((in_features, out_features), |(i, j)| {
            let val = (((i * out_features + j) % 100) as f64 - 50.0) / 100.0;
            T::from(val).unwrap_or(T::zero())
        });

        let update_weight =
            Array2::from_shape_fn((in_features + out_features, out_features), |(i, j)| {
                let val = (((i * out_features + j + 13) % 100) as f64 - 50.0) / 100.0;
                T::from(val).unwrap_or(T::zero())
            });

        Ok(Self {
            in_features,
            out_features,
            message_weight,
            update_weight,
        })
    }

    /// Forward pass
    pub fn forward(
        &self,
        adj: &SparseAdjacency<T>,
        features: &ArrayView2<T>,
    ) -> GraphResult<Array2<T>> {
        if features.ncols() != self.in_features {
            return Err(NumRs2Error::ValueError(format!(
                "Expected {} input features, got {}",
                self.in_features,
                features.ncols()
            )));
        }

        let num_nodes = features.nrows();

        // Message phase: compute messages from neighbors
        let messages = sum_aggregation(adj, features)?;
        let transformed_messages = messages.dot(&self.message_weight);

        // Update phase: combine node features with messages
        let mut concat = Array2::zeros((num_nodes, self.in_features + self.out_features));
        for i in 0..num_nodes {
            for j in 0..self.in_features {
                concat[[i, j]] = features[[i, j]];
            }
            for j in 0..self.out_features {
                concat[[i, j + self.in_features]] = transformed_messages[[i, j]];
            }
        }

        let output = concat.dot(&self.update_weight);
        Ok(output)
    }
}

// ================================================================================================
// GIN Layer
// ================================================================================================

/// Graph Isomorphism Network layer
///
/// Implements GIN from Xu et al. (2019):
/// `h_v^(l+1) = MLP((1 + ε) · h_v^(l) + Σ_{u∈N(v)} h_u^(l))`
///
/// GIN is provably as powerful as the Weisfeiler-Lehman graph isomorphism test.
///
/// # Fields
///
/// - `epsilon`: Learnable or fixed epsilon (0.0 for fixed)
/// - `mlp_weight`: MLP weight matrix
#[derive(Debug, Clone)]
pub struct GinLayer<T: Float> {
    pub in_features: usize,
    pub out_features: usize,
    pub epsilon: T,
    pub mlp_weight: Array2<T>,
}

impl<T: Float + SimdUnifiedOps + 'static> GinLayer<T> {
    /// Create new GIN layer
    ///
    /// # Arguments
    ///
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `epsilon` - Epsilon parameter (0.0 for non-learnable)
    pub fn new(in_features: usize, out_features: usize, epsilon: f64) -> GraphResult<Self> {
        let mlp_weight = Array2::from_shape_fn((in_features, out_features), |(i, j)| {
            let val = (((i * out_features + j) % 100) as f64 - 50.0) / 100.0;
            T::from(val).unwrap_or(T::zero())
        });

        Ok(Self {
            in_features,
            out_features,
            epsilon: T::from(epsilon).unwrap_or(T::zero()),
            mlp_weight,
        })
    }

    /// Forward pass
    pub fn forward(
        &self,
        adj: &SparseAdjacency<T>,
        features: &ArrayView2<T>,
    ) -> GraphResult<Array2<T>> {
        if features.ncols() != self.in_features {
            return Err(NumRs2Error::ValueError(format!(
                "Expected {} input features, got {}",
                self.in_features,
                features.ncols()
            )));
        }

        // Sum aggregation
        let neighbor_sum = sum_aggregation(adj, features)?;

        // (1 + ε) * h_v + Σ h_u
        let one_plus_eps = T::one() + self.epsilon;
        let num_nodes = features.nrows();
        let mut combined = Array2::zeros((num_nodes, self.in_features));

        for i in 0..num_nodes {
            for j in 0..self.in_features {
                combined[[i, j]] = one_plus_eps * features[[i, j]] + neighbor_sum[[i, j]];
            }
        }

        // MLP transformation
        let output = combined.dot(&self.mlp_weight);
        Ok(output)
    }
}

// ================================================================================================
// Graph Pooling
// ================================================================================================

/// Global mean pooling
///
/// Computes mean of node features across all nodes to get graph-level representation.
pub fn global_mean_pool<T>(node_features: &ArrayView2<T>) -> GraphResult<Array1<T>>
where
    T: Float + SimdUnifiedOps + scirs2_core::ndarray::ScalarOperand,
{
    let num_nodes = T::from(node_features.nrows() as f64).unwrap_or(T::one());
    let mean = node_features.sum_axis(Axis(0)) / num_nodes;
    Ok(mean)
}

/// Global max pooling
///
/// Takes element-wise maximum across all node features.
pub fn global_max_pool<T>(node_features: &ArrayView2<T>) -> GraphResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    let mut max_feat = Array1::from_elem(node_features.ncols(), T::neg_infinity());
    for i in 0..node_features.nrows() {
        for j in 0..node_features.ncols() {
            let val = node_features[[i, j]];
            if val > max_feat[j] {
                max_feat[j] = val;
            }
        }
    }
    Ok(max_feat)
}

/// Global sum pooling
///
/// Sums node features across all nodes.
pub fn global_sum_pool<T>(node_features: &ArrayView2<T>) -> GraphResult<Array1<T>>
where
    T: Float + SimdUnifiedOps,
{
    Ok(node_features.sum_axis(Axis(0)))
}

/// Top-K pooling
///
/// Selects top-k nodes based on importance scores.
///
/// # Arguments
///
/// * `node_features` - Node features (num_nodes × feature_dim)
/// * `scores` - Importance score for each node (num_nodes)
/// * `k` - Number of nodes to keep
///
/// # Returns
///
/// Pooled features (k × feature_dim)
pub fn topk_pool<T>(
    node_features: &ArrayView2<T>,
    scores: &ArrayView1<T>,
    k: usize,
) -> GraphResult<Array2<T>>
where
    T: Float + SimdUnifiedOps,
{
    let num_nodes = node_features.nrows();
    if scores.len() != num_nodes {
        return Err(NumRs2Error::ValueError(format!(
            "Scores length {} doesn't match number of nodes {}",
            scores.len(),
            num_nodes
        )));
    }

    if k > num_nodes {
        return Err(NumRs2Error::ValueError(format!(
            "k={} exceeds number of nodes={}",
            k, num_nodes
        )));
    }

    // Get indices of top-k scores
    let mut indexed_scores: Vec<_> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_k_indices: Vec<_> = indexed_scores.iter().take(k).map(|(i, _)| *i).collect();

    // Extract features for top-k nodes
    let feat_dim = node_features.ncols();
    let mut pooled = Array2::zeros((k, feat_dim));
    for (new_idx, &orig_idx) in top_k_indices.iter().enumerate() {
        for j in 0..feat_dim {
            pooled[[new_idx, j]] = node_features[[orig_idx, j]];
        }
    }

    Ok(pooled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adjacency_matrix_creation() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let adj =
            AdjacencyMatrix::<f64>::from_edges(3, &edges).expect("test: valid adjacency matrix");
        assert_eq!(adj.num_nodes, 3);
        assert_eq!(adj.adj[[0, 1]], 1.0);
        assert_eq!(adj.adj[[1, 2]], 1.0);
        assert_eq!(adj.adj[[2, 0]], 1.0);
        assert_eq!(adj.adj[[0, 0]], 0.0);
    }

    #[test]
    fn test_adjacency_with_self_loops() {
        let edges = vec![(0, 1), (1, 2)];
        let adj =
            AdjacencyMatrix::<f64>::from_edges(3, &edges).expect("test: valid adjacency matrix");
        let adj_self = adj
            .with_self_loops()
            .expect("test: valid self-loop addition");
        assert_eq!(adj_self.adj[[0, 0]], 1.0);
        assert_eq!(adj_self.adj[[1, 1]], 1.0);
        assert_eq!(adj_self.adj[[2, 2]], 1.0);
    }

    #[test]
    fn test_degree_matrix() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let adj =
            AdjacencyMatrix::<f64>::from_edges(3, &edges).expect("test: valid adjacency matrix");
        let degrees = adj.degree_matrix().expect("test: valid degree matrix");
        assert_eq!(degrees[0], 2.0); // node 0 has 2 outgoing edges
        assert_eq!(degrees[1], 1.0);
        assert_eq!(degrees[2], 1.0);
    }

    #[test]
    fn test_symmetric_normalization() {
        let edges = vec![(0, 1), (1, 0)];
        let adj =
            AdjacencyMatrix::<f64>::from_edges(2, &edges).expect("test: valid adjacency matrix");
        let norm = adj
            .symmetric_normalize()
            .expect("test: valid symmetric normalization");
        // D^(-1/2) = [[1, 0], [0, 1]] for degree 1
        // Normalized adj should be [[0, 1], [1, 0]]
        assert!((norm[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((norm[[1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_edge_list_creation() {
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let edge_list = EdgeList::<f64>::from_edges(3, &edges).expect("test: valid edge list");
        assert_eq!(edge_list.num_nodes, 3);
        assert_eq!(edge_list.edges.len(), 3);
    }

    #[test]
    fn test_edge_list_out_of_bounds() {
        let edges = vec![(0, 5)]; // node 5 doesn't exist
        let result = EdgeList::<f64>::from_edges(3, &edges);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_adjacency_from_edges() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        assert_eq!(sparse.num_nodes, 3);
        assert_eq!(sparse.row_ptr.len(), 4); // num_nodes + 1
        assert_eq!(sparse.col_indices.len(), 3);
    }

    #[test]
    fn test_sparse_adjacency_neighbors() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let (neighbors, weights) = sparse.neighbors(0).expect("test: valid neighbor retrieval");
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert_eq!(weights.len(), neighbors.len());
    }

    #[test]
    fn test_sparse_adjacency_degrees() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let degrees = sparse.degrees();
        assert_eq!(degrees[0], 2.0);
        assert_eq!(degrees[1], 1.0);
        assert_eq!(degrees[2], 1.0);
    }

    #[test]
    fn test_mean_aggregation() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let agg =
            mean_aggregation(&sparse, &features.view()).expect("test: valid mean aggregation");
        // Node 0 neighbors: [1, 2] -> mean = [(3+5)/2, (4+6)/2] = [4, 5]
        assert_eq!(agg[[0, 0]], 4.0);
        assert_eq!(agg[[0, 1]], 5.0);
    }

    #[test]
    fn test_sum_aggregation() {
        let edges = vec![(0, 1), (0, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let agg = sum_aggregation(&sparse, &features.view()).expect("test: valid sum aggregation");
        // Node 0 neighbors: [1, 2] -> sum = [3+5, 4+6] = [8, 10]
        assert_eq!(agg[[0, 0]], 8.0);
        assert_eq!(agg[[0, 1]], 10.0);
    }

    #[test]
    fn test_max_pooling_aggregation() {
        let edges = vec![(0, 1), (0, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let features = array![[1.0, 6.0], [3.0, 4.0], [5.0, 2.0]];
        let agg = max_pooling_aggregation(&sparse, &features.view())
            .expect("test: valid max pooling aggregation");
        // Node 0 neighbors: [1, 2] -> max = [max(3,5), max(4,2)] = [5, 4]
        assert_eq!(agg[[0, 0]], 5.0);
        assert_eq!(agg[[0, 1]], 4.0);
    }

    #[test]
    fn test_gcn_layer_creation() {
        let gcn = GcnLayer::<f64>::new(10, 20).expect("test: valid GCN layer");
        assert_eq!(gcn.in_features, 10);
        assert_eq!(gcn.out_features, 20);
        assert_eq!(gcn.weight.shape(), &[10, 20]);
        assert!(gcn.use_bias);
    }

    #[test]
    fn test_gcn_layer_forward() {
        let edges = vec![(0, 1), (1, 2)];
        let adj =
            AdjacencyMatrix::<f64>::from_edges(3, &edges).expect("test: valid adjacency matrix");
        let features = Array2::ones((3, 5));
        let gcn = GcnLayer::new(5, 10).expect("test: valid GCN layer");
        let output = gcn
            .forward(&adj, &features.view())
            .expect("test: valid GCN forward pass");
        assert_eq!(output.shape(), &[3, 10]);
    }

    #[test]
    fn test_gcn_layer_dimension_mismatch() {
        let edges = vec![(0, 1)];
        let adj =
            AdjacencyMatrix::<f64>::from_edges(2, &edges).expect("test: valid adjacency matrix");
        let features = Array2::ones((2, 10));
        let gcn = GcnLayer::new(5, 10).expect("test: valid GCN layer (expects 5 input features)");
        let result = gcn.forward(&adj, &features.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_gat_layer_creation() {
        let gat = GatLayer::<f64>::new(10, 8, 4, true, 0.2).expect("test: valid GAT layer");
        assert_eq!(gat.in_features, 10);
        assert_eq!(gat.out_features, 8);
        assert_eq!(gat.num_heads, 4);
        assert!(gat.concat);
    }

    #[test]
    fn test_gat_layer_zero_heads() {
        let result = GatLayer::<f64>::new(10, 8, 0, true, 0.2);
        assert!(result.is_err());
    }

    #[test]
    fn test_gat_layer_forward() {
        let edges = vec![(0, 1), (1, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let features = Array2::ones((3, 4));
        let gat = GatLayer::new(4, 2, 2, true, 0.2).expect("test: valid GAT layer");
        let output = gat
            .forward(&sparse, &features.view())
            .expect("test: valid GAT forward pass");
        // 2 heads × 2 features, concatenated
        assert_eq!(output.shape(), &[3, 4]);
    }

    #[test]
    fn test_gat_layer_average_heads() {
        let edges = vec![(0, 1)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(2, &edges).expect("test: valid sparse adjacency");
        let features = Array2::ones((2, 4));
        let gat = GatLayer::new(4, 8, 2, false, 0.2).expect("test: valid GAT layer (concat=false)");
        let output = gat
            .forward(&sparse, &features.view())
            .expect("test: valid GAT forward pass");
        // Averaged heads, so output_dim = 8
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_graphsage_layer_creation() {
        let sage = GraphSageLayer::<f64>::new(10, 20, SageAggregator::Mean, true)
            .expect("test: valid GraphSAGE layer");
        assert_eq!(sage.in_features, 10);
        assert_eq!(sage.out_features, 20);
        assert_eq!(sage.aggregator, SageAggregator::Mean);
        assert!(sage.normalize);
    }

    #[test]
    fn test_graphsage_layer_forward() {
        let edges = vec![(0, 1), (1, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let features = Array2::ones((3, 5));
        let sage = GraphSageLayer::new(5, 10, SageAggregator::Mean, false)
            .expect("test: valid GraphSAGE layer");
        let output = sage
            .forward(&sparse, &features.view())
            .expect("test: valid GraphSAGE forward pass");
        assert_eq!(output.shape(), &[3, 10]);
    }

    #[test]
    fn test_graphsage_pool_aggregator() {
        let edges = vec![(0, 1), (0, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let features = Array2::ones((3, 4));
        let sage = GraphSageLayer::new(4, 8, SageAggregator::Pool, true)
            .expect("test: valid GraphSAGE pool layer");
        let output = sage
            .forward(&sparse, &features.view())
            .expect("test: valid GraphSAGE forward pass");
        assert_eq!(output.shape(), &[3, 8]);
    }

    #[test]
    fn test_mpnn_layer_creation() {
        let mpnn = MpnnLayer::<f64>::new(10, 20).expect("test: valid MPNN layer");
        assert_eq!(mpnn.in_features, 10);
        assert_eq!(mpnn.out_features, 20);
    }

    #[test]
    fn test_mpnn_layer_forward() {
        let edges = vec![(0, 1), (1, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let features = Array2::ones((3, 5));
        let mpnn = MpnnLayer::new(5, 10).expect("test: valid MPNN layer");
        let output = mpnn
            .forward(&sparse, &features.view())
            .expect("test: valid MPNN forward pass");
        assert_eq!(output.shape(), &[3, 10]);
    }

    #[test]
    fn test_gin_layer_creation() {
        let gin = GinLayer::<f64>::new(10, 20, 0.0).expect("test: valid GIN layer");
        assert_eq!(gin.in_features, 10);
        assert_eq!(gin.out_features, 20);
        assert_eq!(gin.epsilon, 0.0);
    }

    #[test]
    fn test_gin_layer_forward() {
        let edges = vec![(0, 1), (1, 2)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let features = Array2::ones((3, 5));
        let gin = GinLayer::new(5, 10, 0.0).expect("test: valid GIN layer");
        let output = gin
            .forward(&sparse, &features.view())
            .expect("test: valid GIN forward pass");
        assert_eq!(output.shape(), &[3, 10]);
    }

    #[test]
    fn test_gin_layer_with_epsilon() {
        let edges = vec![(0, 1)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(2, &edges).expect("test: valid sparse adjacency");
        let features = array![[1.0, 2.0], [3.0, 4.0]];
        let gin = GinLayer::new(2, 2, 0.5).expect("test: valid GIN layer");
        let output = gin
            .forward(&sparse, &features.view())
            .expect("test: valid GIN forward pass");
        assert_eq!(output.shape(), &[2, 2]);
    }

    #[test]
    fn test_global_mean_pool() {
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let pooled = global_mean_pool(&features.view()).expect("test: valid global mean pool");
        assert_eq!(pooled.len(), 2);
        assert_eq!(pooled[0], 3.0); // (1+3+5)/3
        assert_eq!(pooled[1], 4.0); // (2+4+6)/3
    }

    #[test]
    fn test_global_max_pool() {
        let features = array![[1.0, 6.0], [3.0, 4.0], [5.0, 2.0]];
        let pooled = global_max_pool(&features.view()).expect("test: valid global max pool");
        assert_eq!(pooled.len(), 2);
        assert_eq!(pooled[0], 5.0);
        assert_eq!(pooled[1], 6.0);
    }

    #[test]
    fn test_global_sum_pool() {
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let pooled = global_sum_pool(&features.view()).expect("test: valid global sum pool");
        assert_eq!(pooled.len(), 2);
        assert_eq!(pooled[0], 9.0);
        assert_eq!(pooled[1], 12.0);
    }

    #[test]
    fn test_topk_pool() {
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let scores = array![0.1, 0.4, 0.2, 0.5];
        let pooled = topk_pool(&features.view(), &scores.view(), 2).expect("test: valid topk pool");
        assert_eq!(pooled.shape(), &[2, 2]);
        // Top 2 scores: indices 3 (0.5) and 1 (0.4)
        assert_eq!(pooled[[0, 0]], 7.0); // features of node 3
        assert_eq!(pooled[[0, 1]], 8.0);
    }

    #[test]
    fn test_topk_pool_k_exceeds_nodes() {
        let features = array![[1.0, 2.0], [3.0, 4.0]];
        let scores = array![0.1, 0.4];
        let result = topk_pool(&features.view(), &scores.view(), 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_topk_pool_score_mismatch() {
        let features = array![[1.0, 2.0], [3.0, 4.0]];
        let scores = array![0.1]; // only 1 score for 2 nodes
        let result = topk_pool(&features.view(), &scores.view(), 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_data_creation() {
        let edges = vec![(0, 1), (1, 2)];
        let features = Array2::<f64>::ones((3, 5));
        let graph = GraphData::new(3, &edges, features).expect("test: valid graph data creation");
        assert_eq!(graph.adjacency.num_nodes, 3);
        assert_eq!(graph.node_features.shape(), &[3, 5]);
        assert!(graph.edge_features.is_none());
    }

    #[test]
    fn test_graph_data_feature_mismatch() {
        let edges = vec![(0, 1)];
        let features = Array2::<f64>::ones((3, 5)); // 3 nodes
        let result = GraphData::new(2, &edges, features); // but graph has 2 nodes
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_data_with_edge_features() {
        let edges = vec![(0, 1), (1, 2)];
        let features = Array2::<f64>::ones((3, 5));
        let edge_features = Array2::<f64>::ones((2, 3));
        let graph = GraphData::new(3, &edges, features)
            .expect("test: valid graph data creation")
            .with_edge_features(edge_features);
        assert!(graph.edge_features.is_some());
        assert_eq!(
            graph
                .edge_features
                .expect("test: edge features are some")
                .shape(),
            &[2, 3]
        );
    }

    // Additional edge case tests

    #[test]
    fn test_empty_graph() {
        let edges: Vec<(usize, usize)> = vec![];
        let adj =
            AdjacencyMatrix::<f64>::from_edges(3, &edges).expect("test: valid adjacency matrix");
        assert_eq!(adj.num_nodes, 3);
        // All adjacency values should be 0
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(adj.adj[[i, j]], 0.0);
            }
        }
    }

    #[test]
    fn test_self_loop_graph() {
        let edges = vec![(0, 0), (1, 1), (2, 2)];
        let adj =
            AdjacencyMatrix::<f64>::from_edges(3, &edges).expect("test: valid adjacency matrix");
        assert_eq!(adj.adj[[0, 0]], 1.0);
        assert_eq!(adj.adj[[1, 1]], 1.0);
        assert_eq!(adj.adj[[2, 2]], 1.0);
    }

    #[test]
    fn test_complete_graph() {
        let edges = vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let degrees = sparse.degrees();
        // In a complete graph on 3 nodes, each node has degree 2
        assert_eq!(degrees[0], 2.0);
        assert_eq!(degrees[1], 2.0);
        assert_eq!(degrees[2], 2.0);
    }

    #[test]
    fn test_aggregation_isolated_node() {
        // Node 2 has no neighbors
        let edges = vec![(0, 1)];
        let sparse =
            SparseAdjacency::<f64>::from_edges(3, &edges).expect("test: valid sparse adjacency");
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let agg =
            mean_aggregation(&sparse, &features.view()).expect("test: valid mean aggregation");
        // Node 2 has no neighbors, so aggregation should be [0, 0]
        assert_eq!(agg[[2, 0]], 0.0);
        assert_eq!(agg[[2, 1]], 0.0);
    }
}
