//! Graph Traversal Algorithms
//!
//! This module provides fundamental graph traversal algorithms including BFS, DFS,
//! topological sort, and connected component analysis.
//!
//! # Algorithms
//!
//! - **BFS**: Breadth-first search with distance tracking - O(V + E)
//! - **DFS**: Depth-first search with pre/post ordering - O(V + E)
//! - **Topological Sort**: Kahn's algorithm for DAGs - O(V + E)
//! - **Connected Components**: For undirected graphs - O(V + E)
//! - **Strongly Connected Components**: Tarjan's algorithm - O(V + E)
//! - **Cycle Detection**: For both directed and undirected graphs - O(V + E)

use super::{Graph, GraphError, GraphResult, NodeId};
use std::collections::{HashMap, HashSet, VecDeque};

/// Result of breadth-first search
#[derive(Debug, Clone)]
pub struct BfsResult {
    /// Distance from start node to each reachable node
    pub distances: HashMap<NodeId, usize>,
    /// Predecessor of each node in the BFS tree
    pub predecessors: HashMap<NodeId, Option<NodeId>>,
    /// Nodes in the order they were visited
    pub visited_order: Vec<NodeId>,
}

/// Result of depth-first search
#[derive(Debug, Clone)]
pub struct DfsResult {
    /// Pre-order numbering (when node was first visited)
    pub pre_order: HashMap<NodeId, usize>,
    /// Post-order numbering (when node was finished)
    pub post_order: HashMap<NodeId, usize>,
    /// Predecessor of each node in the DFS tree
    pub predecessors: HashMap<NodeId, Option<NodeId>>,
    /// Nodes in pre-order
    pub pre_order_sequence: Vec<NodeId>,
    /// Nodes in post-order
    pub post_order_sequence: Vec<NodeId>,
}

/// Breadth-first search from a starting node
///
/// Time complexity: O(V + E) where V is vertices, E is edges
/// Space complexity: O(V)
///
/// # Arguments
///
/// * `graph` - The graph to search
/// * `start` - Starting node ID
///
/// # Returns
///
/// BfsResult containing distances, predecessors, and visit order
pub fn bfs(graph: &Graph, start: NodeId) -> GraphResult<BfsResult> {
    if !graph.has_node(start) {
        return Err(GraphError::NodeNotFound(start));
    }

    let mut distances = HashMap::new();
    let mut predecessors = HashMap::new();
    let mut visited_order = Vec::new();
    let mut queue = VecDeque::new();

    distances.insert(start, 0);
    predecessors.insert(start, None);
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        visited_order.push(node);

        let neighbors = graph.neighbors(node)?;
        for &(neighbor, _edge_id) in neighbors {
            if !distances.contains_key(&neighbor) {
                distances.insert(neighbor, distances[&node] + 1);
                predecessors.insert(neighbor, Some(node));
                queue.push_back(neighbor);
            }
        }
    }

    Ok(BfsResult {
        distances,
        predecessors,
        visited_order,
    })
}

/// Depth-first search from a starting node
///
/// Time complexity: O(V + E)
/// Space complexity: O(V)
///
/// # Arguments
///
/// * `graph` - The graph to search
/// * `start` - Starting node ID
///
/// # Returns
///
/// DfsResult containing pre/post-order numbering and sequences
pub fn dfs(graph: &Graph, start: NodeId) -> GraphResult<DfsResult> {
    if !graph.has_node(start) {
        return Err(GraphError::NodeNotFound(start));
    }

    let mut pre_order = HashMap::new();
    let mut post_order = HashMap::new();
    let mut predecessors = HashMap::new();
    let mut pre_order_sequence = Vec::new();
    let mut post_order_sequence = Vec::new();
    let mut pre_counter = 0;
    let mut post_counter = 0;

    dfs_visit(
        graph,
        start,
        None,
        &mut pre_order,
        &mut post_order,
        &mut predecessors,
        &mut pre_order_sequence,
        &mut post_order_sequence,
        &mut pre_counter,
        &mut post_counter,
    )?;

    Ok(DfsResult {
        pre_order,
        post_order,
        predecessors,
        pre_order_sequence,
        post_order_sequence,
    })
}

/// Depth-first search visiting all nodes
///
/// Performs DFS from all unvisited nodes to handle disconnected graphs.
/// Time complexity: O(V + E)
pub fn dfs_full(graph: &Graph) -> GraphResult<DfsResult> {
    let mut pre_order = HashMap::new();
    let mut post_order = HashMap::new();
    let mut predecessors = HashMap::new();
    let mut pre_order_sequence = Vec::new();
    let mut post_order_sequence = Vec::new();
    let mut pre_counter = 0;
    let mut post_counter = 0;

    for &node in &graph.nodes() {
        if !pre_order.contains_key(&node) {
            dfs_visit(
                graph,
                node,
                None,
                &mut pre_order,
                &mut post_order,
                &mut predecessors,
                &mut pre_order_sequence,
                &mut post_order_sequence,
                &mut pre_counter,
                &mut post_counter,
            )?;
        }
    }

    Ok(DfsResult {
        pre_order,
        post_order,
        predecessors,
        pre_order_sequence,
        post_order_sequence,
    })
}

/// Helper function for recursive DFS visit
#[allow(clippy::too_many_arguments)]
fn dfs_visit(
    graph: &Graph,
    node: NodeId,
    predecessor: Option<NodeId>,
    pre_order: &mut HashMap<NodeId, usize>,
    post_order: &mut HashMap<NodeId, usize>,
    predecessors: &mut HashMap<NodeId, Option<NodeId>>,
    pre_order_sequence: &mut Vec<NodeId>,
    post_order_sequence: &mut Vec<NodeId>,
    pre_counter: &mut usize,
    post_counter: &mut usize,
) -> GraphResult<()> {
    pre_order.insert(node, *pre_counter);
    pre_order_sequence.push(node);
    *pre_counter += 1;
    predecessors.insert(node, predecessor);

    let neighbors = graph.neighbors(node)?.to_vec();
    for &(neighbor, _edge_id) in &neighbors {
        if !pre_order.contains_key(&neighbor) {
            dfs_visit(
                graph,
                neighbor,
                Some(node),
                pre_order,
                post_order,
                predecessors,
                pre_order_sequence,
                post_order_sequence,
                pre_counter,
                post_counter,
            )?;
        }
    }

    post_order.insert(node, *post_counter);
    post_order_sequence.push(node);
    *post_counter += 1;

    Ok(())
}

/// Topological sort using Kahn's algorithm
///
/// Only works on directed acyclic graphs (DAGs).
/// Time complexity: O(V + E)
///
/// # Arguments
///
/// * `graph` - The directed graph to sort
///
/// # Returns
///
/// Vector of nodes in topological order
///
/// # Errors
///
/// Returns error if graph contains a cycle or is not directed
pub fn topological_sort(graph: &Graph) -> GraphResult<Vec<NodeId>> {
    if !graph.is_directed() {
        return Err(GraphError::InvalidOperation(
            "Topological sort requires a directed graph".to_string(),
        ));
    }

    let nodes = graph.nodes();
    let mut in_degree: HashMap<NodeId, usize> = HashMap::new();

    // Calculate in-degrees
    for &node in &nodes {
        in_degree.insert(node, graph.in_degree(node)?);
    }

    // Queue of nodes with in-degree 0
    let mut queue: VecDeque<NodeId> = nodes
        .iter()
        .filter(|&&n| in_degree[&n] == 0)
        .copied()
        .collect();

    let mut result = Vec::new();

    while let Some(node) = queue.pop_front() {
        result.push(node);

        let neighbors = graph.neighbors(node)?;
        for &(neighbor, _edge_id) in neighbors {
            let degree = in_degree
                .get_mut(&neighbor)
                .ok_or(GraphError::NodeNotFound(neighbor))?;
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(neighbor);
            }
        }
    }

    // Check if all nodes were processed (no cycle)
    if result.len() != nodes.len() {
        return Err(GraphError::CycleDetected(
            "Graph contains a cycle, topological sort not possible".to_string(),
        ));
    }

    Ok(result)
}

/// Find connected components in an undirected graph
///
/// Time complexity: O(V + E)
///
/// # Arguments
///
/// * `graph` - The undirected graph
///
/// # Returns
///
/// Vector of components, where each component is a vector of node IDs
///
/// # Errors
///
/// Returns error if graph is directed
pub fn connected_components(graph: &Graph) -> GraphResult<Vec<Vec<NodeId>>> {
    if graph.is_directed() {
        return Err(GraphError::InvalidOperation(
            "Connected components requires an undirected graph".to_string(),
        ));
    }

    let nodes = graph.nodes();
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for &node in &nodes {
        if !visited.contains(&node) {
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(node);
            visited.insert(node);

            while let Some(current) = queue.pop_front() {
                component.push(current);
                let neighbors = graph.neighbors(current)?;
                for &(neighbor, _edge_id) in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }

            components.push(component);
        }
    }

    Ok(components)
}

/// Find strongly connected components using Tarjan's algorithm
///
/// Time complexity: O(V + E)
///
/// # Arguments
///
/// * `graph` - The directed graph
///
/// # Returns
///
/// Vector of strongly connected components
pub fn strongly_connected_components(graph: &Graph) -> GraphResult<Vec<Vec<NodeId>>> {
    let mut index_counter = 0;
    let mut stack = Vec::new();
    let mut indices = HashMap::new();
    let mut lowlinks = HashMap::new();
    let mut on_stack = HashSet::new();
    let mut components = Vec::new();

    for &node in &graph.nodes() {
        if !indices.contains_key(&node) {
            tarjan_visit(
                graph,
                node,
                &mut index_counter,
                &mut stack,
                &mut indices,
                &mut lowlinks,
                &mut on_stack,
                &mut components,
            )?;
        }
    }

    Ok(components)
}

/// Helper function for Tarjan's algorithm
#[allow(clippy::too_many_arguments)]
fn tarjan_visit(
    graph: &Graph,
    node: NodeId,
    index_counter: &mut usize,
    stack: &mut Vec<NodeId>,
    indices: &mut HashMap<NodeId, usize>,
    lowlinks: &mut HashMap<NodeId, usize>,
    on_stack: &mut HashSet<NodeId>,
    components: &mut Vec<Vec<NodeId>>,
) -> GraphResult<()> {
    indices.insert(node, *index_counter);
    lowlinks.insert(node, *index_counter);
    *index_counter += 1;
    stack.push(node);
    on_stack.insert(node);

    let neighbors = graph.neighbors(node)?.to_vec();
    for &(neighbor, _edge_id) in &neighbors {
        if !indices.contains_key(&neighbor) {
            tarjan_visit(
                graph,
                neighbor,
                index_counter,
                stack,
                indices,
                lowlinks,
                on_stack,
                components,
            )?;
            let neighbor_lowlink = lowlinks[&neighbor];
            let current_lowlink = lowlinks
                .get_mut(&node)
                .ok_or(GraphError::NodeNotFound(node))?;
            *current_lowlink = (*current_lowlink).min(neighbor_lowlink);
        } else if on_stack.contains(&neighbor) {
            let neighbor_index = indices[&neighbor];
            let current_lowlink = lowlinks
                .get_mut(&node)
                .ok_or(GraphError::NodeNotFound(node))?;
            *current_lowlink = (*current_lowlink).min(neighbor_index);
        }
    }

    // If node is a root of an SCC
    if lowlinks[&node] == indices[&node] {
        let mut component = Vec::new();
        loop {
            let w = stack.pop().ok_or(GraphError::InvalidOperation(
                "Stack underflow in Tarjan's algorithm".to_string(),
            ))?;
            on_stack.remove(&w);
            component.push(w);
            if w == node {
                break;
            }
        }
        components.push(component);
    }

    Ok(())
}

/// Detect if the graph contains a cycle
///
/// Time complexity: O(V + E)
///
/// # Arguments
///
/// * `graph` - The graph to check
///
/// # Returns
///
/// true if the graph contains a cycle, false otherwise
pub fn has_cycle(graph: &Graph) -> GraphResult<bool> {
    if graph.is_directed() {
        has_cycle_directed(graph)
    } else {
        has_cycle_undirected(graph)
    }
}

/// Detect cycle in directed graph using DFS
fn has_cycle_directed(graph: &Graph) -> GraphResult<bool> {
    let nodes = graph.nodes();
    let mut visited = HashSet::new();
    let mut rec_stack = HashSet::new();

    for &node in &nodes {
        if !visited.contains(&node)
            && has_cycle_directed_visit(graph, node, &mut visited, &mut rec_stack)?
        {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Helper for directed cycle detection
fn has_cycle_directed_visit(
    graph: &Graph,
    node: NodeId,
    visited: &mut HashSet<NodeId>,
    rec_stack: &mut HashSet<NodeId>,
) -> GraphResult<bool> {
    visited.insert(node);
    rec_stack.insert(node);

    let neighbors = graph.neighbors(node)?;
    for &(neighbor, _edge_id) in neighbors {
        if !visited.contains(&neighbor) {
            if has_cycle_directed_visit(graph, neighbor, visited, rec_stack)? {
                return Ok(true);
            }
        } else if rec_stack.contains(&neighbor) {
            return Ok(true);
        }
    }

    rec_stack.remove(&node);
    Ok(false)
}

/// Detect cycle in undirected graph using DFS
fn has_cycle_undirected(graph: &Graph) -> GraphResult<bool> {
    let nodes = graph.nodes();
    let mut visited = HashSet::new();

    for &node in &nodes {
        if !visited.contains(&node) && has_cycle_undirected_visit(graph, node, None, &mut visited)?
        {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Helper for undirected cycle detection
fn has_cycle_undirected_visit(
    graph: &Graph,
    node: NodeId,
    parent: Option<NodeId>,
    visited: &mut HashSet<NodeId>,
) -> GraphResult<bool> {
    visited.insert(node);

    let neighbors = graph.neighbors(node)?;
    for &(neighbor, _edge_id) in neighbors {
        if !visited.contains(&neighbor) {
            if has_cycle_undirected_visit(graph, neighbor, Some(node), visited)? {
                return Ok(true);
            }
        } else if Some(neighbor) != parent {
            return Ok(true);
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n0, n2, 1.0).unwrap();
        graph.add_edge(n1, n3, 1.0).unwrap();

        let result = bfs(&graph, n0).unwrap();
        assert_eq!(result.distances[&n0], 0);
        assert_eq!(result.distances[&n1], 1);
        assert_eq!(result.distances[&n2], 1);
        assert_eq!(result.distances[&n3], 2);
    }

    #[test]
    fn test_dfs() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n1, n2, 1.0).unwrap();

        let result = dfs(&graph, n0).unwrap();
        assert!(result.pre_order.contains_key(&n0));
        assert!(result.pre_order.contains_key(&n1));
        assert!(result.pre_order.contains_key(&n2));
        assert_eq!(result.pre_order_sequence.len(), 3);
        assert_eq!(result.post_order_sequence.len(), 3);
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n1, n2, 1.0).unwrap();

        let sorted = topological_sort(&graph).unwrap();
        assert_eq!(sorted.len(), 3);

        // Check topological order property
        let pos: HashMap<_, _> = sorted.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        assert!(pos[&n0] < pos[&n1]);
        assert!(pos[&n1] < pos[&n2]);
    }

    #[test]
    fn test_topological_sort_cycle_detection() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n1, n0, 1.0).unwrap();

        let result = topological_sort(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_connected_components() {
        let mut graph = Graph::new(false);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n2, n3, 1.0).unwrap();

        let components = connected_components(&graph).unwrap();
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_strongly_connected_components() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        graph.add_edge(n1, n2, 1.0).unwrap();
        graph.add_edge(n2, n0, 1.0).unwrap();

        let sccs = strongly_connected_components(&graph).unwrap();
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn test_cycle_detection_directed() {
        let mut graph = Graph::new(true);
        let n0 = graph.add_node();
        let n1 = graph.add_node();

        assert!(!has_cycle(&graph).unwrap());

        graph.add_edge(n0, n1, 1.0).unwrap();
        assert!(!has_cycle(&graph).unwrap());

        graph.add_edge(n1, n0, 1.0).unwrap();
        assert!(has_cycle(&graph).unwrap());
    }

    #[test]
    fn test_cycle_detection_undirected() {
        let mut graph = Graph::new(false);
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1, 1.0).unwrap();
        assert!(!has_cycle(&graph).unwrap());

        graph.add_edge(n1, n2, 1.0).unwrap();
        assert!(!has_cycle(&graph).unwrap());

        graph.add_edge(n2, n0, 1.0).unwrap();
        assert!(has_cycle(&graph).unwrap());
    }
}
