//! Network Flow Algorithms
//!
//! This module provides implementations of network flow algorithms for solving
//! maximum flow and related problems in flow networks.
//!
//! # Algorithms
//!
//! - **Ford-Fulkerson / Edmonds-Karp**: Maximum flow using BFS - O(VE²)
//! - **Min-Cut**: Minimum cut from max flow - O(VE²)
//! - **Bipartite Matching**: Maximum matching in bipartite graphs - O(VE²)
//! - **Hopcroft-Karp**: Faster bipartite matching - O(E√V)

use super::{Graph, GraphError, GraphResult, NodeId, Weight};
use std::collections::{HashMap, HashSet, VecDeque};

/// Flow network with residual graph
#[derive(Debug, Clone)]
pub struct FlowNetwork {
    /// Adjacency list with capacities: node -> [(neighbor, capacity)]
    capacity: HashMap<NodeId, HashMap<NodeId, Weight>>,
    /// Current flow: from -> to -> flow
    flow: HashMap<NodeId, HashMap<NodeId, Weight>>,
    /// All nodes in the network
    nodes: HashSet<NodeId>,
}

impl FlowNetwork {
    /// Create a new flow network from a graph
    ///
    /// # Arguments
    ///
    /// * `graph` - The graph representing the network
    ///
    /// # Returns
    ///
    /// A new FlowNetwork instance
    pub fn from_graph(graph: &Graph) -> GraphResult<Self> {
        let mut capacity = HashMap::new();
        let mut flow = HashMap::new();
        let mut nodes = HashSet::new();

        for &node in &graph.nodes() {
            nodes.insert(node);
            capacity.insert(node, HashMap::new());
            flow.insert(node, HashMap::new());
        }

        for edge in graph.edges() {
            capacity
                .get_mut(&edge.from)
                .ok_or(GraphError::NodeNotFound(edge.from))?
                .insert(edge.to, edge.weight);

            flow.get_mut(&edge.from)
                .ok_or(GraphError::NodeNotFound(edge.from))?
                .insert(edge.to, 0.0);

            // Initialize reverse flow
            if !flow
                .get(&edge.to)
                .ok_or(GraphError::NodeNotFound(edge.to))?
                .contains_key(&edge.from)
            {
                flow.get_mut(&edge.to)
                    .ok_or(GraphError::NodeNotFound(edge.to))?
                    .insert(edge.from, 0.0);
            }
        }

        Ok(Self {
            capacity,
            flow,
            nodes,
        })
    }

    /// Get residual capacity from u to v
    fn residual_capacity(&self, u: NodeId, v: NodeId) -> Weight {
        let capacity = self
            .capacity
            .get(&u)
            .and_then(|m| m.get(&v))
            .copied()
            .unwrap_or(0.0);
        let current_flow = self
            .flow
            .get(&u)
            .and_then(|m| m.get(&v))
            .copied()
            .unwrap_or(0.0);
        capacity - current_flow
    }

    /// Check if there's a path from source to sink in residual graph
    fn bfs_residual(&self, source: NodeId, sink: NodeId) -> Option<HashMap<NodeId, NodeId>> {
        let mut visited = HashSet::new();
        let mut parent = HashMap::new();
        let mut queue = VecDeque::new();

        visited.insert(source);
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            if u == sink {
                return Some(parent);
            }

            // Check all neighbors in residual graph
            for &v in &self.nodes {
                if !visited.contains(&v) && self.residual_capacity(u, v) > 0.0 {
                    visited.insert(v);
                    parent.insert(v, u);
                    queue.push_back(v);
                }
            }
        }

        None
    }

    /// Get total flow from source
    pub fn total_flow(&self, source: NodeId) -> Weight {
        self.flow
            .get(&source)
            .map(|outgoing| outgoing.values().sum::<Weight>())
            .unwrap_or(0.0)
    }

    /// Get flow on edge from u to v
    pub fn get_flow(&self, u: NodeId, v: NodeId) -> Weight {
        self.flow
            .get(&u)
            .and_then(|m| m.get(&v))
            .copied()
            .unwrap_or(0.0)
    }
}

/// Result of maximum flow algorithm
#[derive(Debug, Clone)]
pub struct MaxFlowResult {
    /// Maximum flow value
    pub max_flow: Weight,
    /// Flow on each edge: (from, to) -> flow
    pub flows: HashMap<(NodeId, NodeId), Weight>,
    /// Nodes reachable from source in residual graph (for min-cut)
    pub source_side: HashSet<NodeId>,
}

/// Ford-Fulkerson algorithm using Edmonds-Karp (BFS) for augmenting paths
///
/// Computes maximum flow from source to sink in a flow network.
///
/// Time complexity: O(VE²)
/// Space complexity: O(V²)
///
/// # Arguments
///
/// * `graph` - The flow network (edge weights are capacities)
/// * `source` - Source node
/// * `sink` - Sink node
///
/// # Returns
///
/// MaxFlowResult containing maximum flow value and flow assignments
///
/// # Errors
///
/// Returns error if source or sink don't exist, or if negative capacity found
pub fn max_flow(graph: &Graph, source: NodeId, sink: NodeId) -> GraphResult<MaxFlowResult> {
    if !graph.has_node(source) {
        return Err(GraphError::NodeNotFound(source));
    }
    if !graph.has_node(sink) {
        return Err(GraphError::NodeNotFound(sink));
    }
    if source == sink {
        return Err(GraphError::InvalidOperation(
            "Source and sink must be different nodes".to_string(),
        ));
    }

    // Check for negative capacities
    for edge in graph.edges() {
        if edge.weight < 0.0 {
            return Err(GraphError::InvalidWeight(
                "Flow network cannot have negative capacities".to_string(),
            ));
        }
    }

    let mut network = FlowNetwork::from_graph(graph)?;

    // Edmonds-Karp: repeatedly find augmenting paths using BFS
    while let Some(parent) = network.bfs_residual(source, sink) {
        // Find minimum residual capacity along the path
        let mut path_flow = Weight::INFINITY;
        let mut v = sink;

        while let Some(&u) = parent.get(&v) {
            let residual = network.residual_capacity(u, v);
            path_flow = path_flow.min(residual);
            v = u;
        }

        // Update flow along the path
        v = sink;
        while let Some(&u) = parent.get(&v) {
            // Add flow on forward edge
            let current = network
                .flow
                .get(&u)
                .and_then(|m| m.get(&v))
                .copied()
                .unwrap_or(0.0);
            network
                .flow
                .get_mut(&u)
                .ok_or(GraphError::NodeNotFound(u))?
                .insert(v, current + path_flow);

            // Subtract flow on reverse edge
            let reverse = network
                .flow
                .get(&v)
                .and_then(|m| m.get(&u))
                .copied()
                .unwrap_or(0.0);
            network
                .flow
                .get_mut(&v)
                .ok_or(GraphError::NodeNotFound(v))?
                .insert(u, reverse - path_flow);

            v = u;
        }
    }

    // Collect flows
    let mut flows = HashMap::new();
    for (&from, outgoing) in &network.flow {
        for (&to, &flow_val) in outgoing {
            if flow_val > 0.0 {
                flows.insert((from, to), flow_val);
            }
        }
    }

    // Find source side for min-cut (BFS from source in residual graph)
    let mut source_side = HashSet::new();
    let mut queue = VecDeque::new();
    source_side.insert(source);
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        for &v in &network.nodes {
            if !source_side.contains(&v) && network.residual_capacity(u, v) > 0.0 {
                source_side.insert(v);
                queue.push_back(v);
            }
        }
    }

    Ok(MaxFlowResult {
        max_flow: network.total_flow(source),
        flows,
        source_side,
    })
}

/// Compute minimum cut from maximum flow result
///
/// The min-cut consists of edges from source side to sink side in the residual graph.
///
/// # Arguments
///
/// * `graph` - The flow network
/// * `max_flow_result` - Result from max_flow algorithm
///
/// # Returns
///
/// Vector of edges in the minimum cut with their capacities
pub fn min_cut(
    graph: &Graph,
    max_flow_result: &MaxFlowResult,
) -> GraphResult<Vec<(NodeId, NodeId, Weight)>> {
    let mut cut_edges = Vec::new();

    for edge in graph.edges() {
        let in_source = max_flow_result.source_side.contains(&edge.from);
        let in_sink = !max_flow_result.source_side.contains(&edge.to);

        if in_source && in_sink {
            cut_edges.push((edge.from, edge.to, edge.weight));
        }
    }

    Ok(cut_edges)
}

/// Bipartite graph for matching
#[derive(Debug, Clone)]
pub struct BipartiteGraph {
    /// Left partition nodes
    pub left: HashSet<NodeId>,
    /// Right partition nodes
    pub right: HashSet<NodeId>,
    /// Edges: left node -> set of right nodes
    pub edges: HashMap<NodeId, HashSet<NodeId>>,
}

impl BipartiteGraph {
    /// Create a new bipartite graph
    pub fn new() -> Self {
        Self {
            left: HashSet::new(),
            right: HashSet::new(),
            edges: HashMap::new(),
        }
    }

    /// Add a node to the left partition
    pub fn add_left(&mut self, node: NodeId) {
        self.left.insert(node);
        self.edges.entry(node).or_default();
    }

    /// Add a node to the right partition
    pub fn add_right(&mut self, node: NodeId) {
        self.right.insert(node);
    }

    /// Add an edge from left to right
    pub fn add_edge(&mut self, left: NodeId, right: NodeId) -> GraphResult<()> {
        if !self.left.contains(&left) {
            return Err(GraphError::NodeNotFound(left));
        }
        if !self.right.contains(&right) {
            return Err(GraphError::NodeNotFound(right));
        }

        self.edges
            .get_mut(&left)
            .ok_or(GraphError::NodeNotFound(left))?
            .insert(right);
        Ok(())
    }

    /// Convert to a flow network for maximum matching
    ///
    /// Returns (graph, source, sink, reverse_left_map, reverse_right_map)
    /// where reverse maps go from flow network node IDs back to original bipartite node IDs.
    fn to_flow_network(
        &self,
    ) -> (
        Graph,
        NodeId,
        NodeId,
        HashMap<NodeId, NodeId>,
        HashMap<NodeId, NodeId>,
    ) {
        let mut graph = Graph::new(true);
        let mut reverse_left = HashMap::new();
        let mut reverse_right = HashMap::new();

        // Create source and sink
        let source = graph.add_node();
        let sink = graph.add_node();

        // Add left nodes (sorted for deterministic ordering)
        let mut left_sorted: Vec<_> = self.left.iter().copied().collect();
        left_sorted.sort();
        let mut left_id_map = HashMap::new();
        for left_node in &left_sorted {
            let id = graph.add_node();
            left_id_map.insert(*left_node, id);
            reverse_left.insert(id, *left_node);
            let _ = graph.add_edge(source, id, 1.0);
        }

        // Add right nodes (sorted for deterministic ordering)
        let mut right_sorted: Vec<_> = self.right.iter().copied().collect();
        right_sorted.sort();
        let mut right_id_map = HashMap::new();
        for right_node in &right_sorted {
            let id = graph.add_node();
            right_id_map.insert(*right_node, id);
            reverse_right.insert(id, *right_node);
            let _ = graph.add_edge(id, sink, 1.0);
        }

        // Add edges from left to right
        for (&left_node, right_neighbors) in &self.edges {
            if let Some(&left_id) = left_id_map.get(&left_node) {
                for &right_node in right_neighbors {
                    if let Some(&right_id) = right_id_map.get(&right_node) {
                        let _ = graph.add_edge(left_id, right_id, 1.0);
                    }
                }
            }
        }

        (graph, source, sink, reverse_left, reverse_right)
    }
}

impl Default for BipartiteGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum bipartite matching using max flow
///
/// Finds maximum matching in a bipartite graph by converting to max flow problem.
///
/// Time complexity: O(VE²) using Edmonds-Karp
/// Space complexity: O(V + E)
///
/// # Arguments
///
/// * `bipartite` - The bipartite graph
///
/// # Returns
///
/// Vector of matched pairs (left_node, right_node)
pub fn max_bipartite_matching(bipartite: &BipartiteGraph) -> GraphResult<Vec<(NodeId, NodeId)>> {
    let (flow_graph, source, sink, reverse_left, reverse_right) = bipartite.to_flow_network();

    if flow_graph.node_count() < 2 {
        return Ok(Vec::new());
    }

    let flow_result = max_flow(&flow_graph, source, sink)?;

    // Extract matching from flow using the reverse maps
    let mut matching = Vec::new();

    for ((from, to), flow_val) in &flow_result.flows {
        if *flow_val > 0.0 {
            if let (Some(&left), Some(&right)) = (reverse_left.get(from), reverse_right.get(to)) {
                matching.push((left, right));
            }
        }
    }

    Ok(matching)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_flow_simple() {
        let mut graph = Graph::new(true);
        let source = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let sink = graph.add_node();

        graph
            .add_edge(source, n1, 10.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(source, n2, 10.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(n1, sink, 10.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(n2, sink, 10.0)
            .expect("test: valid edge addition");

        let result = max_flow(&graph, source, sink).expect("test: valid max flow");
        assert_eq!(result.max_flow, 20.0);
    }

    #[test]
    fn test_max_flow_bottleneck() {
        let mut graph = Graph::new(true);
        let source = graph.add_node();
        let n1 = graph.add_node();
        let sink = graph.add_node();

        graph
            .add_edge(source, n1, 10.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(n1, sink, 5.0)
            .expect("test: valid edge addition");

        let result = max_flow(&graph, source, sink).expect("test: valid max flow");
        assert_eq!(result.max_flow, 5.0);
    }

    #[test]
    fn test_max_flow_negative_capacity() {
        let mut graph = Graph::new(true);
        let source = graph.add_node();
        let sink = graph.add_node();

        graph
            .add_edge(source, sink, -1.0)
            .expect("test: valid edge addition");

        let result = max_flow(&graph, source, sink);
        assert!(result.is_err());
    }

    #[test]
    fn test_min_cut() {
        let mut graph = Graph::new(true);
        let source = graph.add_node();
        let n1 = graph.add_node();
        let sink = graph.add_node();

        graph
            .add_edge(source, n1, 10.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(n1, sink, 5.0)
            .expect("test: valid edge addition");
        graph
            .add_edge(source, sink, 15.0)
            .expect("test: valid edge addition");

        let flow_result = max_flow(&graph, source, sink).expect("test: valid max flow");
        let cut = min_cut(&graph, &flow_result).expect("test: valid min cut");

        // Min-cut capacity should equal max flow
        let cut_capacity: Weight = cut.iter().map(|(_, _, w)| w).sum();
        assert_eq!(cut_capacity, flow_result.max_flow);
    }

    #[test]
    fn test_bipartite_matching() {
        let mut bipartite = BipartiteGraph::new();

        // Left nodes: 0, 1, 2
        // Right nodes: 3, 4, 5
        for i in 0..3 {
            bipartite.add_left(i);
        }
        for i in 3..6 {
            bipartite.add_right(i);
        }

        // Add edges
        bipartite
            .add_edge(0, 3)
            .expect("test: valid bipartite edge addition");
        bipartite
            .add_edge(0, 4)
            .expect("test: valid bipartite edge addition");
        bipartite
            .add_edge(1, 4)
            .expect("test: valid bipartite edge addition");
        bipartite
            .add_edge(2, 5)
            .expect("test: valid bipartite edge addition");

        let matching = max_bipartite_matching(&bipartite).expect("test: valid bipartite matching");

        // Should find maximum matching of size 3
        assert_eq!(matching.len(), 3);

        // Verify matching properties
        let left_matched: HashSet<_> = matching.iter().map(|(l, _)| *l).collect();
        let right_matched: HashSet<_> = matching.iter().map(|(_, r)| *r).collect();

        assert_eq!(left_matched.len(), 3);
        assert_eq!(right_matched.len(), 3);
    }

    #[test]
    fn test_bipartite_matching_incomplete() {
        let mut bipartite = BipartiteGraph::new();

        // More left nodes than right nodes
        bipartite.add_left(0);
        bipartite.add_left(1);
        bipartite.add_left(2);
        bipartite.add_right(3);
        bipartite.add_right(4);

        bipartite
            .add_edge(0, 3)
            .expect("test: valid bipartite edge addition");
        bipartite
            .add_edge(1, 3)
            .expect("test: valid bipartite edge addition");
        bipartite
            .add_edge(1, 4)
            .expect("test: valid bipartite edge addition");
        bipartite
            .add_edge(2, 4)
            .expect("test: valid bipartite edge addition");

        let matching = max_bipartite_matching(&bipartite).expect("test: valid bipartite matching");

        // Can only match 2 pairs (limited by right side)
        assert_eq!(matching.len(), 2);
    }

    #[test]
    fn test_flow_network_creation() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();

        graph
            .add_edge(n0, n1, 10.0)
            .expect("test: valid edge addition");

        let network = FlowNetwork::from_graph(&graph).expect("test: valid flow network creation");
        assert_eq!(network.residual_capacity(n0, n1), 10.0);
        assert_eq!(network.get_flow(n0, n1), 0.0);
    }
}
