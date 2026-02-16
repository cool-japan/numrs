//! Shortest Path Algorithms
//!
//! This module provides implementations of classic shortest path algorithms
//! for finding optimal paths in weighted graphs.
//!
//! # Algorithms
//!
//! - **Dijkstra**: Single-source shortest paths (non-negative weights) - O((V + E) log V)
//! - **Bellman-Ford**: Single-source shortest paths (handles negative weights) - O(VE)
//! - **Floyd-Warshall**: All-pairs shortest paths - O(V³)
//! - **A***: Heuristic-based pathfinding - O((V + E) log V) typical case
//!
//! All algorithms include path reconstruction capabilities.

use super::{Graph, GraphError, GraphResult, NodeId, Weight};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Result of single-source shortest path algorithms
#[derive(Debug, Clone)]
pub struct ShortestPathResult {
    /// Distance from source to each reachable node
    pub distances: HashMap<NodeId, Weight>,
    /// Predecessor of each node in shortest path tree
    pub predecessors: HashMap<NodeId, Option<NodeId>>,
}

impl ShortestPathResult {
    /// Reconstruct path from source to target
    ///
    /// Returns None if no path exists.
    pub fn reconstruct_path(&self, target: NodeId) -> Option<Vec<NodeId>> {
        if !self.distances.contains_key(&target) {
            return None;
        }

        let mut path = Vec::new();
        let mut current = Some(target);

        while let Some(node) = current {
            path.push(node);
            current = *self.predecessors.get(&node)?;
        }

        path.reverse();
        Some(path)
    }

    /// Get distance to a node
    pub fn distance_to(&self, node: NodeId) -> Option<Weight> {
        self.distances.get(&node).copied()
    }
}

/// Node for priority queue (min-heap)
#[derive(Debug, Clone, Copy, PartialEq)]
struct PriorityNode {
    node: NodeId,
    distance: Weight,
}

impl Eq for PriorityNode {}

impl PartialOrd for PriorityNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

/// Dijkstra's shortest path algorithm
///
/// Finds shortest paths from a source node to all reachable nodes.
/// Only works with non-negative edge weights.
///
/// Time complexity: O((V + E) log V) with binary heap
/// Space complexity: O(V)
///
/// # Arguments
///
/// * `graph` - The graph to search
/// * `source` - Source node ID
///
/// # Returns
///
/// ShortestPathResult containing distances and predecessors
///
/// # Errors
///
/// Returns error if source node doesn't exist or if negative weights are found
pub fn dijkstra(graph: &Graph, source: NodeId) -> GraphResult<ShortestPathResult> {
    if !graph.has_node(source) {
        return Err(GraphError::NodeNotFound(source));
    }

    let mut distances = HashMap::new();
    let mut predecessors = HashMap::new();
    let mut heap = BinaryHeap::new();

    distances.insert(source, 0.0);
    predecessors.insert(source, None);
    heap.push(PriorityNode {
        node: source,
        distance: 0.0,
    });

    while let Some(PriorityNode { node, distance }) = heap.pop() {
        // Skip if we've already found a better path
        if let Some(&d) = distances.get(&node) {
            if distance > d {
                continue;
            }
        }

        let neighbors = graph.neighbors(node)?;
        for &(neighbor, edge_id) in neighbors {
            let edge = graph.get_edge(edge_id)?;

            if edge.weight < 0.0 {
                return Err(GraphError::InvalidWeight(
                    "Dijkstra's algorithm requires non-negative weights".to_string(),
                ));
            }

            let new_distance = distance + edge.weight;
            let should_update = distances.get(&neighbor).is_none_or(|&d| new_distance < d);

            if should_update {
                distances.insert(neighbor, new_distance);
                predecessors.insert(neighbor, Some(node));
                heap.push(PriorityNode {
                    node: neighbor,
                    distance: new_distance,
                });
            }
        }
    }

    Ok(ShortestPathResult {
        distances,
        predecessors,
    })
}

/// Bellman-Ford shortest path algorithm
///
/// Finds shortest paths from a source node to all reachable nodes.
/// Can handle negative edge weights and detects negative cycles.
///
/// Time complexity: O(VE)
/// Space complexity: O(V)
///
/// # Arguments
///
/// * `graph` - The graph to search
/// * `source` - Source node ID
///
/// # Returns
///
/// ShortestPathResult containing distances and predecessors
///
/// # Errors
///
/// Returns error if source node doesn't exist or if a negative cycle is detected
pub fn bellman_ford(graph: &Graph, source: NodeId) -> GraphResult<ShortestPathResult> {
    if !graph.has_node(source) {
        return Err(GraphError::NodeNotFound(source));
    }

    let nodes = graph.nodes();
    let mut distances = HashMap::new();
    let mut predecessors = HashMap::new();

    // Initialize distances
    for &node in &nodes {
        distances.insert(node, Weight::INFINITY);
        predecessors.insert(node, None);
    }
    distances.insert(source, 0.0);

    // Relax edges V-1 times
    for _ in 0..nodes.len() - 1 {
        let mut updated = false;
        for edge in graph.edges() {
            let dist_from = distances[&edge.from];
            if dist_from.is_finite() {
                let new_distance = dist_from + edge.weight;
                if new_distance < distances[&edge.to] {
                    distances.insert(edge.to, new_distance);
                    predecessors.insert(edge.to, Some(edge.from));
                    updated = true;
                }
            }
        }
        if !updated {
            break;
        }
    }

    // Check for negative cycles
    for edge in graph.edges() {
        let dist_from = distances[&edge.from];
        if dist_from.is_finite() {
            let new_distance = dist_from + edge.weight;
            if new_distance < distances[&edge.to] {
                return Err(GraphError::NegativeCycle(
                    "Graph contains a negative cycle".to_string(),
                ));
            }
        }
    }

    Ok(ShortestPathResult {
        distances,
        predecessors,
    })
}

/// Result of all-pairs shortest path algorithm
#[derive(Debug, Clone)]
pub struct AllPairsShortestPaths {
    /// Distance matrix: distances\[i\]\[j\] = distance from node i to node j
    pub distances: HashMap<NodeId, HashMap<NodeId, Weight>>,
    /// Next node matrix for path reconstruction
    pub next: HashMap<NodeId, HashMap<NodeId, Option<NodeId>>>,
}

impl AllPairsShortestPaths {
    /// Reconstruct path from source to target
    ///
    /// Returns None if no path exists.
    pub fn reconstruct_path(&self, source: NodeId, target: NodeId) -> Option<Vec<NodeId>> {
        let dist = self.distances.get(&source)?.get(&target)?;
        if dist.is_infinite() {
            return None;
        }

        let mut path = vec![source];
        let mut current = source;

        while current != target {
            let next_opt = self.next.get(&current)?.get(&target)?;
            let next_node = (*next_opt)?;
            path.push(next_node);
            current = next_node;
        }

        Some(path)
    }

    /// Get distance between two nodes
    pub fn distance(&self, source: NodeId, target: NodeId) -> Option<Weight> {
        self.distances.get(&source)?.get(&target).copied()
    }
}

/// Floyd-Warshall all-pairs shortest paths algorithm
///
/// Computes shortest paths between all pairs of nodes.
/// Can handle negative weights but not negative cycles.
///
/// Time complexity: O(V³)
/// Space complexity: O(V²)
///
/// # Arguments
///
/// * `graph` - The graph to analyze
///
/// # Returns
///
/// AllPairsShortestPaths containing distance and next matrices
///
/// # Errors
///
/// Returns error if a negative cycle is detected
pub fn floyd_warshall(graph: &Graph) -> GraphResult<AllPairsShortestPaths> {
    let nodes = graph.nodes();
    let mut distances: HashMap<NodeId, HashMap<NodeId, Weight>> = HashMap::new();
    let mut next: HashMap<NodeId, HashMap<NodeId, Option<NodeId>>> = HashMap::new();

    // Initialize distances
    for &i in &nodes {
        let mut row_dist = HashMap::new();
        let mut row_next = HashMap::new();
        for &j in &nodes {
            if i == j {
                row_dist.insert(j, 0.0);
                row_next.insert(j, None);
            } else {
                row_dist.insert(j, Weight::INFINITY);
                row_next.insert(j, None);
            }
        }
        distances.insert(i, row_dist);
        next.insert(i, row_next);
    }

    // Set edge weights
    for edge in graph.edges() {
        distances
            .get_mut(&edge.from)
            .ok_or(GraphError::NodeNotFound(edge.from))?
            .insert(edge.to, edge.weight);
        next.get_mut(&edge.from)
            .ok_or(GraphError::NodeNotFound(edge.from))?
            .insert(edge.to, Some(edge.to));
    }

    // Floyd-Warshall main loop
    for &k in &nodes {
        for &i in &nodes {
            for &j in &nodes {
                let dist_ik = distances[&i][&k];
                let dist_kj = distances[&k][&j];
                let dist_ij = distances[&i][&j];

                if dist_ik.is_finite() && dist_kj.is_finite() {
                    let new_dist = dist_ik + dist_kj;
                    if new_dist < dist_ij {
                        distances
                            .get_mut(&i)
                            .ok_or(GraphError::NodeNotFound(i))?
                            .insert(j, new_dist);
                        let next_k = next[&k][&j];
                        next.get_mut(&i)
                            .ok_or(GraphError::NodeNotFound(i))?
                            .insert(j, next_k);
                    }
                }
            }
        }
    }

    // Check for negative cycles
    for &i in &nodes {
        if distances[&i][&i] < 0.0 {
            return Err(GraphError::NegativeCycle(format!(
                "Negative cycle detected involving node {}",
                i
            )));
        }
    }

    Ok(AllPairsShortestPaths { distances, next })
}

/// Heuristic function type for A* search
///
/// Takes a node and target, returns estimated distance to target.
pub type HeuristicFn = Box<dyn Fn(NodeId, NodeId) -> Weight>;

/// A* pathfinding algorithm
///
/// Finds shortest path from source to target using a heuristic function.
/// Heuristic must be admissible (never overestimate) for optimality.
///
/// Time complexity: O((V + E) log V) typical case (depends on heuristic)
/// Space complexity: O(V)
///
/// # Arguments
///
/// * `graph` - The graph to search
/// * `source` - Source node ID
/// * `target` - Target node ID
/// * `heuristic` - Admissible heuristic function
///
/// # Returns
///
/// Path from source to target and total cost
///
/// # Errors
///
/// Returns error if source or target don't exist, or if no path exists
pub fn astar(
    graph: &Graph,
    source: NodeId,
    target: NodeId,
    heuristic: HeuristicFn,
) -> GraphResult<(Vec<NodeId>, Weight)> {
    if !graph.has_node(source) {
        return Err(GraphError::NodeNotFound(source));
    }
    if !graph.has_node(target) {
        return Err(GraphError::NodeNotFound(target));
    }

    let mut g_score = HashMap::new();
    let mut f_score = HashMap::new();
    let mut predecessors = HashMap::new();
    let mut heap = BinaryHeap::new();

    g_score.insert(source, 0.0);
    f_score.insert(source, heuristic(source, target));
    heap.push(PriorityNode {
        node: source,
        distance: heuristic(source, target),
    });

    while let Some(PriorityNode { node, distance: _ }) = heap.pop() {
        if node == target {
            // Reconstruct path
            let mut path = vec![target];
            let mut current = target;
            while let Some(&pred) = predecessors.get(&current) {
                path.push(pred);
                current = pred;
            }
            path.reverse();
            return Ok((path, g_score[&target]));
        }

        let neighbors = graph.neighbors(node)?;
        for &(neighbor, edge_id) in neighbors {
            let edge = graph.get_edge(edge_id)?;
            let tentative_g_score = g_score[&node] + edge.weight;

            let should_update = g_score
                .get(&neighbor)
                .is_none_or(|&g| tentative_g_score < g);

            if should_update {
                predecessors.insert(neighbor, node);
                g_score.insert(neighbor, tentative_g_score);
                let f = tentative_g_score + heuristic(neighbor, target);
                f_score.insert(neighbor, f);
                heap.push(PriorityNode {
                    node: neighbor,
                    distance: f,
                });
            }
        }
    }

    Err(GraphError::InvalidPath(format!(
        "No path exists from {} to {}",
        source, target
    )))
}

/// Reconstruct path from predecessors map
///
/// # Arguments
///
/// * `predecessors` - Map of node to predecessor
/// * `target` - Target node
///
/// # Returns
///
/// Path from source to target, or None if no path exists
pub fn reconstruct_path(
    predecessors: &HashMap<NodeId, Option<NodeId>>,
    target: NodeId,
) -> Option<Vec<NodeId>> {
    if !predecessors.contains_key(&target) {
        return None;
    }

    let mut path = Vec::new();
    let mut current = Some(target);

    while let Some(node) = current {
        path.push(node);
        current = *predecessors.get(&node)?;
    }

    path.reverse();
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n0, n2, 4.0).unwrap();
        graph.add_edge(n1, n2, 2.0).unwrap();
        graph.add_edge(n1, n3, 5.0).unwrap();
        graph.add_edge(n2, n3, 1.0).unwrap();

        graph
    }

    #[test]
    fn test_dijkstra() {
        let graph = create_test_graph();
        let result = dijkstra(&graph, 0).unwrap();

        assert_eq!(result.distance_to(0), Some(0.0));
        assert_eq!(result.distance_to(1), Some(1.0));
        assert_eq!(result.distance_to(2), Some(3.0));
        assert_eq!(result.distance_to(3), Some(4.0));

        let path = result.reconstruct_path(3).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_dijkstra_negative_weight() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        graph.add_edge(n0, n1, -1.0).unwrap();

        let result = dijkstra(&graph, n0);
        assert!(result.is_err());
    }

    #[test]
    fn test_bellman_ford() {
        let graph = create_test_graph();
        let result = bellman_ford(&graph, 0).unwrap();

        assert_eq!(result.distance_to(0), Some(0.0));
        assert_eq!(result.distance_to(1), Some(1.0));
        assert_eq!(result.distance_to(2), Some(3.0));
        assert_eq!(result.distance_to(3), Some(4.0));
    }

    #[test]
    fn test_bellman_ford_negative_weights() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n1, n2, -2.0).unwrap();

        let result = bellman_ford(&graph, n0).unwrap();
        assert_eq!(result.distance_to(2), Some(-1.0));
    }

    #[test]
    fn test_bellman_ford_negative_cycle() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n1, n0, -2.0).unwrap();

        let result = bellman_ford(&graph, n0);
        assert!(result.is_err());
    }

    #[test]
    fn test_floyd_warshall() {
        let graph = create_test_graph();
        let result = floyd_warshall(&graph).unwrap();

        assert_eq!(result.distance(0, 3), Some(4.0));
        assert_eq!(result.distance(0, 0), Some(0.0));

        let path = result.reconstruct_path(0, 3).unwrap();
        assert_eq!(path[0], 0);
        assert_eq!(path[path.len() - 1], 3);
    }

    #[test]
    fn test_astar() {
        let graph = create_test_graph();

        // Zero heuristic (equivalent to Dijkstra)
        let heuristic = Box::new(|_: NodeId, _: NodeId| 0.0);
        let (path, cost) = astar(&graph, 0, 3, heuristic).unwrap();

        assert_eq!(cost, 4.0);
        assert_eq!(path[0], 0);
        assert_eq!(path[path.len() - 1], 3);
    }

    #[test]
    fn test_astar_no_path() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        // No edge between them

        let heuristic = Box::new(|_: NodeId, _: NodeId| 0.0);
        let result = astar(&graph, n0, n1, heuristic);
        assert!(result.is_err());
    }

    #[test]
    fn test_reconstruct_path() {
        let mut predecessors = HashMap::new();
        predecessors.insert(0, None);
        predecessors.insert(1, Some(0));
        predecessors.insert(2, Some(1));

        let path = reconstruct_path(&predecessors, 2).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
    }
}
