//! Graph Algorithms Module
//!
//! This module provides comprehensive graph data structures and algorithms for NumRS2.
//! It includes efficient implementations of traversal, shortest path, spanning tree,
//! and network flow algorithms.
//!
//! # Features
//!
//! - **Graph Representation**: Adjacency list-based graphs (directed/undirected, weighted/unweighted)
//! - **Traversal**: BFS, DFS, topological sort, connected components, cycle detection
//! - **Shortest Paths**: Dijkstra, Bellman-Ford, Floyd-Warshall, A*
//! - **Spanning Trees**: Kruskal, Prim, Borůvka algorithms with Union-Find
//! - **Network Flow**: Max flow, min cut, bipartite matching
//!
//! # Examples
//!
//! ```
//! use numrs2::new_modules::graph::{Graph, NodeId};
//!
//! // Create a directed graph
//! let mut graph = Graph::new(true);
//! let n0 = graph.add_node();
//! let n1 = graph.add_node();
//! graph.add_edge(n0, n1, 1.0).unwrap();
//! ```
//!
//! # SCIRS2 Integration
//!
//! This module follows NumRS2's SCIRS2 integration policy:
//! - Uses `scirs2_core::ndarray` for array operations
//! - No direct external dependencies (rand, ndarray, rayon)
//! - Pure Rust implementation via SciRS2 ecosystem

use crate::error::NumRs2Error;
use std::collections::HashMap;
use std::fmt;

// Submodules
pub mod flow;
pub mod shortest_path;
pub mod spanning_tree;
pub mod traversal;

// Re-exports
pub use flow::*;
pub use shortest_path::*;
pub use spanning_tree::*;
pub use traversal::*;

/// Node identifier in the graph
pub type NodeId = usize;

/// Edge identifier in the graph
pub type EdgeId = usize;

/// Edge weight type
pub type Weight = f64;

/// Result type for graph operations
pub type GraphResult<T> = Result<T, GraphError>;

/// Graph-specific error types
#[derive(Debug, Clone)]
pub enum GraphError {
    /// Node does not exist in the graph
    NodeNotFound(NodeId),

    /// Edge does not exist in the graph
    EdgeNotFound(EdgeId),

    /// Invalid operation on graph
    InvalidOperation(String),

    /// Graph contains a cycle (when cycle-free is required)
    CycleDetected(String),

    /// Graph is not connected
    NotConnected(String),

    /// Negative cycle detected
    NegativeCycle(String),

    /// Invalid path
    InvalidPath(String),

    /// Graph is empty
    EmptyGraph,

    /// Invalid weight value
    InvalidWeight(String),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::NodeNotFound(id) => write!(f, "Node {} not found in graph", id),
            GraphError::EdgeNotFound(id) => write!(f, "Edge {} not found in graph", id),
            GraphError::InvalidOperation(msg) => write!(f, "Invalid graph operation: {}", msg),
            GraphError::CycleDetected(msg) => write!(f, "Cycle detected: {}", msg),
            GraphError::NotConnected(msg) => write!(f, "Graph is not connected: {}", msg),
            GraphError::NegativeCycle(msg) => write!(f, "Negative cycle detected: {}", msg),
            GraphError::InvalidPath(msg) => write!(f, "Invalid path: {}", msg),
            GraphError::EmptyGraph => write!(f, "Graph is empty"),
            GraphError::InvalidWeight(msg) => write!(f, "Invalid weight: {}", msg),
        }
    }
}

impl std::error::Error for GraphError {}

/// Convert GraphError to NumRs2Error
impl From<GraphError> for NumRs2Error {
    fn from(err: GraphError) -> Self {
        NumRs2Error::InvalidOperation(err.to_string())
    }
}

/// Edge in the graph
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    /// Edge identifier
    pub id: EdgeId,
    /// Source node
    pub from: NodeId,
    /// Target node
    pub to: NodeId,
    /// Edge weight
    pub weight: Weight,
}

impl Edge {
    /// Create a new edge
    pub fn new(id: EdgeId, from: NodeId, to: NodeId, weight: Weight) -> Self {
        Self {
            id,
            from,
            to,
            weight,
        }
    }
}

/// Graph data structure using adjacency list representation
///
/// Supports both directed and undirected graphs with weighted edges.
/// Time complexity for common operations:
/// - Add node: O(1)
/// - Add edge: O(1) average
/// - Get neighbors: O(1)
/// - Remove node: O(V + E) where V is nodes, E is edges
#[derive(Debug, Clone)]
pub struct Graph {
    /// Whether the graph is directed
    directed: bool,
    /// Adjacency list: node -> list of (neighbor, edge_id)
    adj_list: HashMap<NodeId, Vec<(NodeId, EdgeId)>>,
    /// Edge storage: edge_id -> Edge
    edges: HashMap<EdgeId, Edge>,
    /// Next node ID to assign
    next_node_id: NodeId,
    /// Next edge ID to assign
    next_edge_id: EdgeId,
    /// Node count
    node_count: usize,
}

impl Graph {
    /// Create a new graph
    ///
    /// # Arguments
    ///
    /// * `directed` - Whether the graph is directed
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::new_modules::graph::Graph;
    ///
    /// let graph = Graph::new(true);  // Directed graph
    /// let ugraph = Graph::new(false); // Undirected graph
    /// ```
    pub fn new(directed: bool) -> Self {
        Self {
            directed,
            adj_list: HashMap::new(),
            edges: HashMap::new(),
            next_node_id: 0,
            next_edge_id: 0,
            node_count: 0,
        }
    }

    /// Add a node to the graph
    ///
    /// Returns the ID of the newly created node.
    /// Time complexity: O(1)
    pub fn add_node(&mut self) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.adj_list.insert(id, Vec::new());
        self.node_count += 1;
        id
    }

    /// Add an edge to the graph
    ///
    /// # Arguments
    ///
    /// * `from` - Source node ID
    /// * `to` - Target node ID
    /// * `weight` - Edge weight
    ///
    /// Returns the ID of the newly created edge.
    /// Time complexity: O(1) average
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, weight: Weight) -> GraphResult<EdgeId> {
        if !self.has_node(from) {
            return Err(GraphError::NodeNotFound(from));
        }
        if !self.has_node(to) {
            return Err(GraphError::NodeNotFound(to));
        }

        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;

        let edge = Edge::new(edge_id, from, to, weight);
        self.edges.insert(edge_id, edge);

        // Add to adjacency list
        self.adj_list
            .get_mut(&from)
            .ok_or(GraphError::NodeNotFound(from))?
            .push((to, edge_id));

        // For undirected graphs, add reverse edge
        if !self.directed {
            self.adj_list
                .get_mut(&to)
                .ok_or(GraphError::NodeNotFound(to))?
                .push((from, edge_id));
        }

        Ok(edge_id)
    }

    /// Check if a node exists in the graph
    pub fn has_node(&self, node: NodeId) -> bool {
        self.adj_list.contains_key(&node)
    }

    /// Check if an edge exists in the graph
    pub fn has_edge(&self, edge_id: EdgeId) -> bool {
        self.edges.contains_key(&edge_id)
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Get the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if the graph is directed
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> Vec<NodeId> {
        self.adj_list.keys().copied().collect()
    }

    /// Get neighbors of a node
    ///
    /// Returns a list of (neighbor_id, edge_id) pairs.
    /// Time complexity: O(1)
    pub fn neighbors(&self, node: NodeId) -> GraphResult<&[(NodeId, EdgeId)]> {
        self.adj_list
            .get(&node)
            .map(|v| v.as_slice())
            .ok_or(GraphError::NodeNotFound(node))
    }

    /// Get an edge by ID
    pub fn get_edge(&self, edge_id: EdgeId) -> GraphResult<&Edge> {
        self.edges
            .get(&edge_id)
            .ok_or(GraphError::EdgeNotFound(edge_id))
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> Vec<&Edge> {
        self.edges.values().collect()
    }

    /// Get the degree (number of neighbors) of a node
    pub fn degree(&self, node: NodeId) -> GraphResult<usize> {
        Ok(self.neighbors(node)?.len())
    }

    /// Get the in-degree of a node (for directed graphs)
    ///
    /// For undirected graphs, this is the same as degree.
    pub fn in_degree(&self, node: NodeId) -> GraphResult<usize> {
        if !self.has_node(node) {
            return Err(GraphError::NodeNotFound(node));
        }

        if !self.directed {
            return self.degree(node);
        }

        let count = self.edges.values().filter(|e| e.to == node).count();
        Ok(count)
    }

    /// Get the out-degree of a node (for directed graphs)
    ///
    /// For undirected graphs, this is the same as degree.
    pub fn out_degree(&self, node: NodeId) -> GraphResult<usize> {
        self.degree(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let directed = Graph::new(true);
        assert!(directed.is_directed());
        assert_eq!(directed.node_count(), 0);
        assert_eq!(directed.edge_count(), 0);

        let undirected = Graph::new(false);
        assert!(!undirected.is_directed());
    }

    #[test]
    fn test_add_nodes() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        assert_eq!(n0, 0);
        assert_eq!(n1, 1);
        assert_eq!(n2, 2);
        assert_eq!(graph.node_count(), 3);
        assert!(graph.has_node(0));
        assert!(graph.has_node(1));
        assert!(graph.has_node(2));
        assert!(!graph.has_node(3));
    }

    #[test]
    fn test_add_edges_directed() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();

        let e0 = graph.add_edge(n0, n1, 1.5).unwrap();
        assert_eq!(graph.edge_count(), 1);

        let edge = graph.get_edge(e0).unwrap();
        assert_eq!(edge.from, n0);
        assert_eq!(edge.to, n1);
        assert_eq!(edge.weight, 1.5);

        let neighbors = graph.neighbors(n0).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, n1);
    }

    #[test]
    fn test_add_edges_undirected() {
        let mut graph = Graph::new(false);
        let n0 = graph.add_node();
        let n1 = graph.add_node();

        graph.add_edge(n0, n1, 2.0).unwrap();

        // Both directions should have the edge
        let neighbors0 = graph.neighbors(n0).unwrap();
        let neighbors1 = graph.neighbors(n1).unwrap();
        assert_eq!(neighbors0.len(), 1);
        assert_eq!(neighbors1.len(), 1);
        assert_eq!(neighbors0[0].0, n1);
        assert_eq!(neighbors1[0].0, n0);
    }

    #[test]
    fn test_degree() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n0, n2, 1.0).unwrap();
        graph.add_edge(n1, n2, 1.0).unwrap();

        assert_eq!(graph.out_degree(n0).unwrap(), 2);
        assert_eq!(graph.in_degree(n0).unwrap(), 0);
        assert_eq!(graph.out_degree(n1).unwrap(), 1);
        assert_eq!(graph.in_degree(n1).unwrap(), 1);
        assert_eq!(graph.out_degree(n2).unwrap(), 0);
        assert_eq!(graph.in_degree(n2).unwrap(), 2);
    }

    #[test]
    fn test_error_handling() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();

        // Try to add edge with non-existent node
        assert!(graph.add_edge(n0, 999, 1.0).is_err());
        assert!(graph.add_edge(999, n0, 1.0).is_err());

        // Try to get neighbors of non-existent node
        assert!(graph.neighbors(999).is_err());
    }
}
