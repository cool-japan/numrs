//! # Wavelet Packet Decomposition
//!
//! This module implements wavelet packet decomposition, which extends the DWT
//! by decomposing both approximation AND detail coefficients at each level,
//! creating a full binary tree of decompositions.
//!
//! ## Wavelet Packet Tree
//!
//! Unlike DWT which only decomposes approximations:
//!
//! ```text
//! DWT:           x
//!               / \
//!             a₁  d₁
//!             /\
//!           a₂ d₂
//! ```
//!
//! Wavelet packets decompose both branches:
//!
//! ```text
//! WPT:           x
//!               / \
//!             a₁  d₁
//!            / \  / \
//!          a₂ d₂ a₂ d₂
//! ```
//!
//! ## Best Basis Selection
//!
//! The wavelet packet tree provides many possible bases for signal representation.
//! The "best basis" minimizes an entropy measure, providing optimal compression
//! or denoising.

use super::{dwt_1d, idwt_1d, ExtensionMode, Wavelet, WaveletError, WaveletResult};
use std::collections::HashMap;

/// Enumeration of entropy criteria for best basis selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BestBasisCriterion {
    /// Shannon entropy: -Σ p_i log(p_i)
    Shannon,
    /// Log energy entropy: Σ log(|x_i|²)
    LogEnergy,
    /// Threshold entropy: count of coefficients above threshold
    Threshold,
    /// L¹ norm (sum of absolute values)
    L1Norm,
}

impl BestBasisCriterion {
    /// Compute entropy of a coefficient set
    pub fn compute(&self, coeffs: &[f64], threshold: f64) -> f64 {
        match self {
            BestBasisCriterion::Shannon => {
                let energy: f64 = coeffs.iter().map(|x| x * x).sum();
                if energy == 0.0 {
                    return 0.0;
                }

                let mut entropy = 0.0;
                for &x in coeffs {
                    let p = (x * x) / energy;
                    if p > 0.0 {
                        entropy -= p * p.ln();
                    }
                }
                entropy
            }
            BestBasisCriterion::LogEnergy => {
                let mut entropy = 0.0;
                for &x in coeffs {
                    let x2 = x * x;
                    if x2 > 0.0 {
                        entropy += x2.ln();
                    }
                }
                entropy
            }
            BestBasisCriterion::Threshold => {
                coeffs.iter().filter(|&&x| x.abs() > threshold).count() as f64
            }
            BestBasisCriterion::L1Norm => coeffs.iter().map(|x| x.abs()).sum(),
        }
    }
}

/// Node identifier in the wavelet packet tree
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId {
    /// Decomposition level (0 = root)
    pub level: usize,
    /// Position at this level (0 to 2^level - 1)
    pub index: usize,
}

impl NodeId {
    /// Create a new node identifier
    pub fn new(level: usize, index: usize) -> Self {
        Self { level, index }
    }

    /// Get the root node (level 0, index 0)
    pub fn root() -> Self {
        Self { level: 0, index: 0 }
    }

    /// Get the left child node
    pub fn left_child(&self) -> Self {
        Self {
            level: self.level + 1,
            index: 2 * self.index,
        }
    }

    /// Get the right child node
    pub fn right_child(&self) -> Self {
        Self {
            level: self.level + 1,
            index: 2 * self.index + 1,
        }
    }

    /// Get the parent node
    pub fn parent(&self) -> Option<Self> {
        if self.level == 0 {
            None
        } else {
            Some(Self {
                level: self.level - 1,
                index: self.index / 2,
            })
        }
    }

    /// Check if this is a leaf node in a tree of given depth
    pub fn is_leaf(&self, max_level: usize) -> bool {
        self.level == max_level
    }
}

/// Wavelet packet tree node
#[derive(Debug, Clone)]
pub struct WaveletPacketNode {
    /// Node identifier
    pub id: NodeId,
    /// Coefficients at this node
    pub coefficients: Vec<f64>,
    /// Entropy/cost of this node
    pub cost: f64,
}

/// Wavelet packet decomposition tree
pub struct WaveletPacket {
    /// All nodes in the tree
    nodes: HashMap<NodeId, WaveletPacketNode>,
    /// Maximum decomposition level
    max_level: usize,
    /// Best basis nodes (selected by cost minimization)
    best_basis: Vec<NodeId>,
}

impl WaveletPacket {
    /// Create a new wavelet packet decomposition
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            max_level: 0,
            best_basis: Vec::new(),
        }
    }

    /// Get a node by its identifier
    pub fn get_node(&self, id: NodeId) -> Option<&WaveletPacketNode> {
        self.nodes.get(&id)
    }

    /// Get the maximum decomposition level
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Get the best basis node identifiers
    pub fn best_basis(&self) -> &[NodeId] {
        &self.best_basis
    }

    /// Get all leaf nodes at maximum level
    pub fn leaf_nodes(&self) -> Vec<NodeId> {
        let mut leaves = Vec::new();
        for level_idx in 0..(1 << self.max_level) {
            let node_id = NodeId::new(self.max_level, level_idx);
            if self.nodes.contains_key(&node_id) {
                leaves.push(node_id);
            }
        }
        leaves
    }

    /// Reconstruct signal from selected nodes
    pub fn reconstruct(
        &self,
        nodes: &[NodeId],
        wavelet: &dyn Wavelet,
        original_len: usize,
    ) -> WaveletResult<Vec<f64>> {
        if nodes.is_empty() {
            return Err(WaveletError::InsufficientData(
                "No nodes provided for reconstruction".to_string(),
            ));
        }

        // Start with the coefficients from selected nodes
        let mut level_nodes: HashMap<usize, Vec<(usize, Vec<f64>)>> = HashMap::new();

        for &node_id in nodes {
            let node = self.get_node(node_id).ok_or_else(|| {
                WaveletError::ComputationError(format!("Node {:?} not found", node_id))
            })?;

            level_nodes
                .entry(node_id.level)
                .or_default()
                .push((node_id.index, node.coefficients.clone()));
        }

        // Reconstruct level by level from deepest to shallowest
        let mut current_level = self.max_level;

        while current_level > 0 {
            let prev_level = current_level - 1;

            if let Some(nodes_at_level) = level_nodes.get(&current_level) {
                let mut parent_nodes: HashMap<usize, (Vec<f64>, Vec<f64>)> = HashMap::new();

                for (index, coeffs) in nodes_at_level {
                    let parent_idx = index / 2;
                    let is_left = index % 2 == 0;

                    let entry = parent_nodes
                        .entry(parent_idx)
                        .or_insert_with(|| (Vec::new(), Vec::new()));

                    if is_left {
                        entry.0 = coeffs.clone();
                    } else {
                        entry.1 = coeffs.clone();
                    }
                }

                // Reconstruct parent coefficients
                for (parent_idx, (left, right)) in parent_nodes {
                    if !left.is_empty() && !right.is_empty() {
                        let output_len = left.len() * 2;
                        let reconstructed = idwt_1d(&left, &right, wavelet, output_len)?;

                        level_nodes
                            .entry(prev_level)
                            .or_default()
                            .push((parent_idx, reconstructed));
                    }
                }
            }

            current_level -= 1;
        }

        // Get the root reconstruction
        if let Some(root_nodes) = level_nodes.get(&0) {
            if let Some((_, coeffs)) = root_nodes.first() {
                let mut result = coeffs.clone();
                result.truncate(original_len);
                return Ok(result);
            }
        }

        Err(WaveletError::ComputationError(
            "Failed to reconstruct signal".to_string(),
        ))
    }
}

impl Default for WaveletPacket {
    fn default() -> Self {
        Self::new()
    }
}

/// Perform wavelet packet decomposition
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `wavelet` - Wavelet to use
/// * `level` - Maximum decomposition level
/// * `mode` - Boundary extension mode
///
/// # Returns
///
/// Wavelet packet tree with all nodes computed
///
/// # Examples
///
/// ```rust,ignore
/// use numrs::new_modules::wavelets::{packet_decompose, WaveletType, ExtensionMode};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let wavelet = WaveletType::Haar.create()?;
/// let wpt = packet_decompose(&signal, &wavelet, 3, ExtensionMode::Periodic)?;
/// ```
pub fn packet_decompose(
    signal: &[f64],
    wavelet: &dyn Wavelet,
    level: usize,
    mode: ExtensionMode,
) -> WaveletResult<WaveletPacket> {
    if signal.is_empty() {
        return Err(WaveletError::InvalidLength(
            "Signal must not be empty".to_string(),
        ));
    }

    if level == 0 {
        return Err(WaveletError::InvalidLevel(
            "Level must be at least 1".to_string(),
        ));
    }

    let mut wpt = WaveletPacket::new();
    wpt.max_level = level;

    // Initialize root node
    let root_id = NodeId::root();
    wpt.nodes.insert(
        root_id,
        WaveletPacketNode {
            id: root_id,
            coefficients: signal.to_vec(),
            cost: 0.0,
        },
    );

    // Decompose level by level
    for current_level in 0..level {
        let num_nodes = 1 << current_level;

        for node_idx in 0..num_nodes {
            let node_id = NodeId::new(current_level, node_idx);

            if let Some(node) = wpt.nodes.get(&node_id).cloned() {
                // Decompose this node
                let (approx, detail) = dwt_1d(&node.coefficients, wavelet, mode)?;

                // Create child nodes
                let left_id = node_id.left_child();
                let right_id = node_id.right_child();

                wpt.nodes.insert(
                    left_id,
                    WaveletPacketNode {
                        id: left_id,
                        coefficients: approx,
                        cost: 0.0,
                    },
                );

                wpt.nodes.insert(
                    right_id,
                    WaveletPacketNode {
                        id: right_id,
                        coefficients: detail,
                        cost: 0.0,
                    },
                );
            }
        }
    }

    Ok(wpt)
}

/// Select best basis using entropy criterion
///
/// # Arguments
///
/// * `wpt` - Wavelet packet tree
/// * `criterion` - Entropy criterion to minimize
/// * `threshold` - Threshold for threshold-based criteria
///
/// # Returns
///
/// Modified wavelet packet with best_basis field set
pub fn select_best_basis(
    wpt: &mut WaveletPacket,
    criterion: BestBasisCriterion,
    threshold: f64,
) -> WaveletResult<()> {
    // Compute costs for all nodes
    let node_ids: Vec<NodeId> = wpt.nodes.keys().copied().collect();

    for node_id in node_ids {
        if let Some(node) = wpt.nodes.get(&node_id) {
            let cost = criterion.compute(&node.coefficients, threshold);
            if let Some(node) = wpt.nodes.get_mut(&node_id) {
                node.cost = cost;
            }
        }
    }

    // Select best basis using dynamic programming (bottom-up)
    let mut best_basis = Vec::new();

    for current_level in (0..=wpt.max_level).rev() {
        let num_nodes = 1 << current_level;

        for node_idx in 0..num_nodes {
            let node_id = NodeId::new(current_level, node_idx);

            if current_level == wpt.max_level {
                // Leaf nodes are always in consideration
                best_basis.push(node_id);
            } else {
                // Compare node cost with sum of children costs
                let left_child = node_id.left_child();
                let right_child = node_id.right_child();

                let node_cost = wpt
                    .nodes
                    .get(&node_id)
                    .map(|n| n.cost)
                    .unwrap_or(f64::INFINITY);

                let left_cost = wpt
                    .nodes
                    .get(&left_child)
                    .map(|n| n.cost)
                    .unwrap_or(f64::INFINITY);

                let right_cost = wpt
                    .nodes
                    .get(&right_child)
                    .map(|n| n.cost)
                    .unwrap_or(f64::INFINITY);

                let children_cost = left_cost + right_cost;

                // Choose node if it's cheaper than decomposing
                if node_cost <= children_cost {
                    // Remove children from best basis if present
                    best_basis.retain(|&id| id != left_child && id != right_child);
                    best_basis.push(node_id);
                }
            }
        }
    }

    wpt.best_basis = best_basis;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_modules::wavelets::WaveletType;

    #[test]
    fn test_node_id_operations() {
        let root = NodeId::root();
        assert_eq!(root.level, 0);
        assert_eq!(root.index, 0);
        assert!(root.parent().is_none());

        let left = root.left_child();
        assert_eq!(left.level, 1);
        assert_eq!(left.index, 0);

        let right = root.right_child();
        assert_eq!(right.level, 1);
        assert_eq!(right.index, 1);

        assert_eq!(left.parent(), Some(root));
        assert_eq!(right.parent(), Some(root));
    }

    #[test]
    fn test_node_id_is_leaf() {
        let node = NodeId::new(2, 3);
        assert!(node.is_leaf(2));
        assert!(!node.is_leaf(3));
    }

    #[test]
    fn test_best_basis_criterion_shannon() {
        let coeffs = vec![1.0, 2.0, 3.0, 4.0];
        let entropy = BestBasisCriterion::Shannon.compute(&coeffs, 0.0);
        assert!(entropy >= 0.0);
    }

    #[test]
    fn test_best_basis_criterion_log_energy() {
        let coeffs = vec![1.0, 2.0, 3.0];
        let entropy = BestBasisCriterion::LogEnergy.compute(&coeffs, 0.0);
        assert!(entropy.is_finite());
    }

    #[test]
    fn test_best_basis_criterion_threshold() {
        let coeffs = vec![0.1, 0.5, 1.0, 2.0];
        let count = BestBasisCriterion::Threshold.compute(&coeffs, 0.4);
        assert_eq!(count, 3.0);
    }

    #[test]
    fn test_best_basis_criterion_l1_norm() {
        let coeffs = vec![-1.0, 2.0, -3.0, 4.0];
        let norm = BestBasisCriterion::L1Norm.compute(&coeffs, 0.0);
        assert_eq!(norm, 10.0);
    }

    #[test]
    fn test_packet_decompose() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let wpt = packet_decompose(&signal, wavelet.as_ref(), 2, ExtensionMode::Periodic)
            .expect("Packet decomposition failed");

        assert_eq!(wpt.max_level(), 2);

        // Check root node
        let root = wpt.get_node(NodeId::root());
        assert!(root.is_some());

        // Check level 1 nodes
        assert!(wpt.get_node(NodeId::new(1, 0)).is_some());
        assert!(wpt.get_node(NodeId::new(1, 1)).is_some());

        // Check level 2 nodes
        assert!(wpt.get_node(NodeId::new(2, 0)).is_some());
        assert!(wpt.get_node(NodeId::new(2, 1)).is_some());
        assert!(wpt.get_node(NodeId::new(2, 2)).is_some());
        assert!(wpt.get_node(NodeId::new(2, 3)).is_some());
    }

    #[test]
    fn test_packet_decompose_empty_signal() {
        let signal: Vec<f64> = vec![];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let result = packet_decompose(&signal, wavelet.as_ref(), 2, ExtensionMode::Periodic);
        assert!(result.is_err());
    }

    #[test]
    fn test_packet_decompose_zero_level() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let result = packet_decompose(&signal, wavelet.as_ref(), 0, ExtensionMode::Periodic);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_best_basis() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mut wpt = packet_decompose(&signal, wavelet.as_ref(), 2, ExtensionMode::Periodic)
            .expect("Packet decomposition failed");

        select_best_basis(&mut wpt, BestBasisCriterion::Shannon, 0.0)
            .expect("Best basis selection failed");

        let best_basis = wpt.best_basis();
        assert!(!best_basis.is_empty());
    }

    #[test]
    fn test_wavelet_packet_reconstruction() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let mut wpt = packet_decompose(&signal, wavelet.as_ref(), 2, ExtensionMode::Periodic)
            .expect("Packet decomposition failed");

        select_best_basis(&mut wpt, BestBasisCriterion::Shannon, 0.0)
            .expect("Best basis selection failed");

        let reconstructed = wpt
            .reconstruct(wpt.best_basis(), wavelet.as_ref(), signal.len())
            .expect("Reconstruction failed");

        assert_eq!(reconstructed.len(), signal.len());
    }

    #[test]
    fn test_leaf_nodes() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let wavelet = WaveletType::Haar
            .create()
            .expect("Failed to create wavelet");

        let wpt = packet_decompose(&signal, wavelet.as_ref(), 2, ExtensionMode::Periodic)
            .expect("Packet decomposition failed");

        let leaves = wpt.leaf_nodes();
        assert_eq!(leaves.len(), 4);

        for leaf in leaves {
            assert_eq!(leaf.level, 2);
        }
    }
}
