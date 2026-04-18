//! Minimum Spanning Tree Algorithms
//!
//! This module provides implementations of classic minimum spanning tree (MST)
//! algorithms for undirected weighted graphs.
//!
//! # Algorithms
//!
//! - **Kruskal**: MST using union-find (edge-based) - O(E log E)
//! - **Prim**: MST using priority queue (vertex-based) - O((V + E) log V)
//! - **Borůvka**: MST using parallel edge selection - O(E log V)
//!
//! All algorithms use the Union-Find data structure with path compression
//! and union by rank for optimal performance.

use super::{Edge, EdgeId, Graph, GraphError, GraphResult, NodeId, Weight};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Union-Find (Disjoint Set Union) data structure
///
/// Supports efficient union and find operations with path compression
/// and union by rank optimization.
///
/// Time complexity:
/// - find: O(α(n)) amortized, where α is inverse Ackermann function
/// - union: O(α(n)) amortized
#[derive(Debug, Clone)]
pub struct UnionFind {
    /// Parent of each element
    parent: HashMap<NodeId, NodeId>,
    /// Rank (approximate tree height) of each set
    rank: HashMap<NodeId, usize>,
}

impl UnionFind {
    /// Create a new Union-Find structure
    pub fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
        }
    }

    /// Make a new set containing a single element
    pub fn make_set(&mut self, x: NodeId) {
        if let std::collections::hash_map::Entry::Vacant(e) = self.parent.entry(x) {
            e.insert(x);
            self.rank.insert(x, 0);
        }
    }

    /// Find the representative (root) of the set containing x
    ///
    /// Uses path compression for optimization.
    pub fn find(&mut self, x: NodeId) -> NodeId {
        if self.parent[&x] != x {
            let root = self.find(self.parent[&x]);
            self.parent.insert(x, root);
        }
        self.parent[&x]
    }

    /// Union the sets containing x and y
    ///
    /// Returns true if x and y were in different sets (union performed),
    /// false if they were already in the same set.
    ///
    /// Uses union by rank for optimization.
    pub fn union(&mut self, x: NodeId, y: NodeId) -> bool {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return false;
        }

        let rank_x = self.rank[&root_x];
        let rank_y = self.rank[&root_y];

        // Attach smaller rank tree under root of higher rank tree
        if rank_x < rank_y {
            self.parent.insert(root_x, root_y);
        } else if rank_x > rank_y {
            self.parent.insert(root_y, root_x);
        } else {
            self.parent.insert(root_y, root_x);
            self.rank.insert(root_x, rank_x + 1);
        }

        true
    }

    /// Check if x and y are in the same set
    pub fn connected(&mut self, x: NodeId, y: NodeId) -> bool {
        self.find(x) == self.find(y)
    }
}

impl Default for UnionFind {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of minimum spanning tree algorithm
#[derive(Debug, Clone)]
pub struct MstResult {
    /// Edges in the minimum spanning tree
    pub edges: Vec<EdgeId>,
    /// Total weight of the spanning tree
    pub total_weight: Weight,
}

impl MstResult {
    /// Get the edges in the MST
    pub fn edge_ids(&self) -> &[EdgeId] {
        &self.edges
    }

    /// Get the total weight
    pub fn weight(&self) -> Weight {
        self.total_weight
    }
}

/// Edge for priority queue (min-heap)
#[derive(Debug, Clone, Copy, PartialEq)]
struct PriorityEdge {
    edge_id: EdgeId,
    weight: Weight,
}

impl Eq for PriorityEdge {}

impl PartialOrd for PriorityEdge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEdge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .weight
            .partial_cmp(&self.weight)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.edge_id.cmp(&other.edge_id))
    }
}

/// Kruskal's minimum spanning tree algorithm
///
/// Finds MST by sorting edges by weight and greedily adding edges
/// that don't create cycles (using Union-Find).
///
/// Time complexity: O(E log E)
/// Space complexity: O(V + E)
///
/// # Arguments
///
/// * `graph` - The undirected graph
///
/// # Returns
///
/// MstResult containing MST edges and total weight
///
/// # Errors
///
/// Returns error if graph is directed or empty
pub fn kruskal(graph: &Graph) -> GraphResult<MstResult> {
    if graph.is_directed() {
        return Err(GraphError::InvalidOperation(
            "Kruskal's algorithm requires an undirected graph".to_string(),
        ));
    }

    if graph.node_count() == 0 {
        return Err(GraphError::EmptyGraph);
    }

    let mut union_find = UnionFind::new();
    for &node in &graph.nodes() {
        union_find.make_set(node);
    }

    // Sort edges by weight
    let mut edges: Vec<&Edge> = graph.edges();
    edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(Ordering::Equal));

    let mut mst_edges = Vec::new();
    let mut total_weight = 0.0;

    for edge in edges {
        // For undirected graphs, check if nodes are not already connected
        if union_find.union(edge.from, edge.to) {
            mst_edges.push(edge.id);
            total_weight += edge.weight;

            // MST has V-1 edges
            if mst_edges.len() == graph.node_count() - 1 {
                break;
            }
        }
    }

    if mst_edges.len() != graph.node_count() - 1 {
        return Err(GraphError::NotConnected(
            "Graph is not connected, no spanning tree exists".to_string(),
        ));
    }

    Ok(MstResult {
        edges: mst_edges,
        total_weight,
    })
}

/// Prim's minimum spanning tree algorithm
///
/// Finds MST by starting from a node and greedily adding the minimum
/// weight edge that connects the tree to a new vertex.
///
/// Time complexity: O((V + E) log V) with binary heap
/// Space complexity: O(V)
///
/// # Arguments
///
/// * `graph` - The undirected graph
/// * `start` - Optional starting node (uses any node if None)
///
/// # Returns
///
/// MstResult containing MST edges and total weight
///
/// # Errors
///
/// Returns error if graph is directed, empty, or not connected
pub fn prim(graph: &Graph, start: Option<NodeId>) -> GraphResult<MstResult> {
    if graph.is_directed() {
        return Err(GraphError::InvalidOperation(
            "Prim's algorithm requires an undirected graph".to_string(),
        ));
    }

    if graph.node_count() == 0 {
        return Err(GraphError::EmptyGraph);
    }

    let start_node = start.unwrap_or_else(|| graph.nodes()[0]);
    if !graph.has_node(start_node) {
        return Err(GraphError::NodeNotFound(start_node));
    }

    let mut in_mst = HashSet::new();
    let mut mst_edges = Vec::new();
    let mut total_weight = 0.0;
    let mut heap = BinaryHeap::new();

    // Add starting node to MST
    in_mst.insert(start_node);

    // Add all edges from start node to heap
    let neighbors = graph.neighbors(start_node)?;
    for &(_neighbor, edge_id) in neighbors {
        let edge = graph.get_edge(edge_id)?;
        heap.push(PriorityEdge {
            edge_id,
            weight: edge.weight,
        });
    }

    while let Some(PriorityEdge { edge_id, weight }) = heap.pop() {
        let edge = graph.get_edge(edge_id)?;

        // Determine which node is new (not in MST)
        let new_node = if in_mst.contains(&edge.from) && !in_mst.contains(&edge.to) {
            Some(edge.to)
        } else if in_mst.contains(&edge.to) && !in_mst.contains(&edge.from) {
            Some(edge.from)
        } else {
            None
        };

        if let Some(node) = new_node {
            // Add edge to MST
            mst_edges.push(edge_id);
            total_weight += weight;
            in_mst.insert(node);

            // MST complete
            if in_mst.len() == graph.node_count() {
                break;
            }

            // Add edges from new node
            let neighbors = graph.neighbors(node)?;
            for &(neighbor, neighbor_edge_id) in neighbors {
                if !in_mst.contains(&neighbor) {
                    let neighbor_edge = graph.get_edge(neighbor_edge_id)?;
                    heap.push(PriorityEdge {
                        edge_id: neighbor_edge_id,
                        weight: neighbor_edge.weight,
                    });
                }
            }
        }
    }

    if mst_edges.len() != graph.node_count() - 1 {
        return Err(GraphError::NotConnected(
            "Graph is not connected, no spanning tree exists".to_string(),
        ));
    }

    Ok(MstResult {
        edges: mst_edges,
        total_weight,
    })
}

/// Borůvka's minimum spanning tree algorithm
///
/// Finds MST by repeatedly finding the minimum weight edge from each
/// component and merging components.
///
/// Time complexity: O(E log V)
/// Space complexity: O(V + E)
///
/// # Arguments
///
/// * `graph` - The undirected graph
///
/// # Returns
///
/// MstResult containing MST edges and total weight
///
/// # Errors
///
/// Returns error if graph is directed, empty, or not connected
pub fn boruvka(graph: &Graph) -> GraphResult<MstResult> {
    if graph.is_directed() {
        return Err(GraphError::InvalidOperation(
            "Borůvka's algorithm requires an undirected graph".to_string(),
        ));
    }

    if graph.node_count() == 0 {
        return Err(GraphError::EmptyGraph);
    }

    let mut union_find = UnionFind::new();
    for &node in &graph.nodes() {
        union_find.make_set(node);
    }

    let mut mst_edges = Vec::new();
    let mut total_weight = 0.0;
    let target_edges = graph.node_count() - 1;

    while mst_edges.len() < target_edges {
        // Find minimum edge for each component
        let mut min_edge: HashMap<NodeId, Option<(EdgeId, Weight)>> = HashMap::new();

        for edge in graph.edges() {
            let comp_from = union_find.find(edge.from);
            let comp_to = union_find.find(edge.to);

            if comp_from != comp_to {
                // Update minimum edge for comp_from
                let should_update_from = min_edge
                    .get(&comp_from)
                    .and_then(|e| e.as_ref())
                    .is_none_or(|(_, w)| edge.weight < *w);

                if should_update_from {
                    min_edge.insert(comp_from, Some((edge.id, edge.weight)));
                }

                // Update minimum edge for comp_to
                let should_update_to = min_edge
                    .get(&comp_to)
                    .and_then(|e| e.as_ref())
                    .is_none_or(|(_, w)| edge.weight < *w);

                if should_update_to {
                    min_edge.insert(comp_to, Some((edge.id, edge.weight)));
                }
            }
        }

        // Check if any edges were found
        if min_edge.values().all(|e| e.is_none()) {
            break;
        }

        // Add minimum edges and merge components
        let mut added_edges = HashSet::new();
        for (_component, edge_info) in min_edge {
            if let Some((edge_id, weight)) = edge_info {
                if added_edges.contains(&edge_id) {
                    continue;
                }

                let edge = graph.get_edge(edge_id)?;
                if union_find.union(edge.from, edge.to) {
                    mst_edges.push(edge_id);
                    total_weight += weight;
                    added_edges.insert(edge_id);

                    if mst_edges.len() >= target_edges {
                        break;
                    }
                }
            }
        }

        // Prevent infinite loop if no progress
        if added_edges.is_empty() {
            break;
        }
    }

    if mst_edges.len() != target_edges {
        return Err(GraphError::NotConnected(
            "Graph is not connected, no spanning tree exists".to_string(),
        ));
    }

    Ok(MstResult {
        edges: mst_edges,
        total_weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new(false);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();

        graph
            .add_edge(n0, n1, 1.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(n0, n2, 4.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(n1, n2, 2.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(n1, n3, 5.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(n2, n3, 3.0)
            .expect("test: valid edge addition");

        graph
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new();
        uf.make_set(0);
        uf.make_set(1);
        uf.make_set(2);

        assert!(!uf.connected(0, 1));
        assert!(uf.union(0, 1));
        assert!(uf.connected(0, 1));

        assert!(!uf.union(0, 1)); // Already connected
        assert!(uf.union(1, 2));
        assert!(uf.connected(0, 2));
    }

    #[test]
    fn test_kruskal() {
        let graph = create_test_graph();
        let result = kruskal(&graph).expect("test: valid Kruskal MST");

        assert_eq!(result.edges.len(), 3); // V-1 edges
        assert_eq!(result.total_weight, 6.0); // 1 + 2 + 3

        // Verify all edges have correct weights
        let mut weights: Vec<Weight> = result
            .edges
            .iter()
            .map(|&e| graph.get_edge(e).expect("test: valid edge").weight)
            .collect();
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(weights, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_kruskal_directed_graph() {
        let graph = Graph::new(true);
        let result = kruskal(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_kruskal_disconnected() {
        let mut graph = Graph::new(false);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();

        graph
            .add_edge(n0, n1, 1.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(n2, n3, 1.0)
            .expect("test: valid edge addition");
        // Two separate components

        let result = kruskal(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_prim() {
        let graph = create_test_graph();
        let result = prim(&graph, None).expect("test: valid Prim MST");

        assert_eq!(result.edges.len(), 3); // V-1 edges
        assert_eq!(result.total_weight, 6.0);
    }

    #[test]
    fn test_prim_with_start() {
        let graph = create_test_graph();
        let result = prim(&graph, Some(2)).expect("test: valid Prim MST with start");

        assert_eq!(result.edges.len(), 3);
        assert_eq!(result.total_weight, 6.0);
    }

    #[test]
    fn test_prim_disconnected() {
        let mut graph = Graph::new(false);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph
            .add_edge(n0, n1, 1.0)
            .expect("test: valid edge addition");
        // n2 is disconnected

        let result = prim(&graph, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_boruvka() {
        let graph = create_test_graph();
        let result = boruvka(&graph).expect("test: valid Boruvka MST");

        assert_eq!(result.edges.len(), 3); // V-1 edges
        assert_eq!(result.total_weight, 6.0);
    }

    #[test]
    fn test_boruvka_disconnected() {
        let mut graph = Graph::new(false);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph
            .add_edge(n0, n1, 1.0)
            .expect("test: valid edge addition");
        // n2 is disconnected

        let result = boruvka(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_mst_algorithms_consistency() {
        let graph = create_test_graph();

        let kruskal_result = kruskal(&graph).expect("test: valid Kruskal MST");
        let prim_result = prim(&graph, None).expect("test: valid Prim MST");
        let boruvka_result = boruvka(&graph).expect("test: valid Boruvka MST");

        // All should have same total weight
        assert_eq!(kruskal_result.total_weight, prim_result.total_weight);
        assert_eq!(prim_result.total_weight, boruvka_result.total_weight);

        // All should have same number of edges
        assert_eq!(kruskal_result.edges.len(), prim_result.edges.len());
        assert_eq!(prim_result.edges.len(), boruvka_result.edges.len());
    }
}
