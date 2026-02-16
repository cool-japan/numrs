//! Boundary condition handling for the FEM module.
//!
//! Supports Dirichlet (essential), Neumann (natural), and penalty-based
//! boundary conditions for scalar and vector FEM problems.
//!
//! # Boundary Condition Types
//!
//! - **Dirichlet**: Prescribed values (e.g., temperature, displacement)
//! - **Neumann**: Prescribed flux/traction on the boundary
//! - **Penalty**: Large-number penalty method for constraint enforcement
//!
//! # Application Methods
//!
//! Boundary conditions are applied to the assembled global system using:
//! - Row/column elimination for Dirichlet BCs
//! - Load vector modification for Neumann BCs
//! - Penalty terms added to stiffness and load for penalty method

use super::assembly::{CsrMatrix, GlobalSystem};
use super::mesh::Mesh;
use super::{FemError, FemResult};
use std::collections::HashMap;

/// A Dirichlet (essential) boundary condition
///
/// Prescribes the value of the solution at specific nodes.
/// For heat equation: prescribed temperature.
/// For elasticity: prescribed displacement.
#[derive(Clone, Debug)]
pub struct DirichletBc {
    /// Node index where the BC is applied
    pub node_id: usize,
    /// Local DOF index (0 for scalar, 0 or 1 for 2D vector)
    pub local_dof: usize,
    /// Prescribed value
    pub value: f64,
}

impl DirichletBc {
    /// Creates a new Dirichlet BC
    ///
    /// # Arguments
    /// * `node_id` - Node where the condition is applied
    /// * `local_dof` - Local DOF index within the node
    /// * `value` - Prescribed value
    pub fn new(node_id: usize, local_dof: usize, value: f64) -> Self {
        Self {
            node_id,
            local_dof,
            value,
        }
    }

    /// Creates a scalar Dirichlet BC (local_dof = 0)
    pub fn scalar(node_id: usize, value: f64) -> Self {
        Self::new(node_id, 0, value)
    }
}

/// A Neumann (natural) boundary condition
///
/// Prescribes the flux or traction on the boundary.
/// For heat equation: prescribed heat flux q.
/// For elasticity: prescribed traction.
#[derive(Clone, Debug)]
pub struct NeumannBc {
    /// Node index where the BC is applied
    pub node_id: usize,
    /// Local DOF index
    pub local_dof: usize,
    /// Prescribed flux/force value
    pub value: f64,
}

impl NeumannBc {
    /// Creates a new Neumann BC
    ///
    /// # Arguments
    /// * `node_id` - Node where the condition is applied
    /// * `local_dof` - Local DOF index
    /// * `value` - Prescribed flux/force
    pub fn new(node_id: usize, local_dof: usize, value: f64) -> Self {
        Self {
            node_id,
            local_dof,
            value,
        }
    }

    /// Creates a scalar Neumann BC (local_dof = 0)
    pub fn scalar(node_id: usize, value: f64) -> Self {
        Self::new(node_id, 0, value)
    }
}

/// Collection of boundary conditions for a FEM problem
#[derive(Clone, Debug, Default)]
pub struct BoundaryConditionSet {
    /// Dirichlet boundary conditions
    pub dirichlet: Vec<DirichletBc>,
    /// Neumann boundary conditions
    pub neumann: Vec<NeumannBc>,
    /// Penalty factor for penalty method (if used)
    pub penalty_factor: Option<f64>,
}

impl BoundaryConditionSet {
    /// Creates an empty boundary condition set
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a Dirichlet boundary condition
    pub fn add_dirichlet(&mut self, bc: DirichletBc) -> &mut Self {
        self.dirichlet.push(bc);
        self
    }

    /// Adds a Neumann boundary condition
    pub fn add_neumann(&mut self, bc: NeumannBc) -> &mut Self {
        self.neumann.push(bc);
        self
    }

    /// Sets the penalty factor for penalty method enforcement
    pub fn set_penalty_factor(&mut self, factor: f64) -> &mut Self {
        self.penalty_factor = Some(factor);
        self
    }

    /// Adds Dirichlet BCs for a scalar problem from (node_id, value) pairs
    pub fn add_scalar_dirichlet(&mut self, conditions: &[(usize, f64)]) -> &mut Self {
        for &(node_id, value) in conditions {
            self.dirichlet.push(DirichletBc::scalar(node_id, value));
        }
        self
    }

    /// Adds a point force (Neumann BC) at a node
    pub fn add_point_force(&mut self, node_id: usize, dof: usize, force: f64) -> &mut Self {
        self.neumann.push(NeumannBc::new(node_id, dof, force));
        self
    }

    /// Returns the number of Dirichlet conditions
    pub fn num_dirichlet(&self) -> usize {
        self.dirichlet.len()
    }

    /// Returns the number of Neumann conditions
    pub fn num_neumann(&self) -> usize {
        self.neumann.len()
    }

    /// Returns true if the set has no conditions
    pub fn is_empty(&self) -> bool {
        self.dirichlet.is_empty() && self.neumann.is_empty()
    }
}

/// Represents a complete set of boundary conditions ready to apply
#[derive(Clone, Debug)]
pub struct BoundaryCondition {
    /// Map of constrained global DOF -> prescribed value
    pub constrained_dofs: HashMap<usize, f64>,
    /// Map of DOFs with Neumann contributions -> force value
    pub neumann_loads: HashMap<usize, f64>,
    /// Penalty factor (if using penalty method)
    pub penalty_factor: Option<f64>,
}

impl BoundaryCondition {
    /// Processes the BoundaryConditionSet into global DOF-level conditions
    ///
    /// # Arguments
    /// * `bc_set` - The boundary condition specification
    /// * `mesh` - The mesh
    /// * `dofs_per_node` - Number of DOFs per node
    pub fn from_set(
        bc_set: &BoundaryConditionSet,
        mesh: &Mesh,
        dofs_per_node: usize,
    ) -> FemResult<Self> {
        let mut constrained_dofs = HashMap::new();
        let mut neumann_loads = HashMap::new();

        // Process Dirichlet BCs
        for bc in &bc_set.dirichlet {
            if bc.node_id >= mesh.num_nodes() {
                return Err(FemError::BoundaryError(format!(
                    "Dirichlet BC node {} out of range (mesh has {} nodes)",
                    bc.node_id,
                    mesh.num_nodes()
                )));
            }
            if bc.local_dof >= dofs_per_node {
                return Err(FemError::BoundaryError(format!(
                    "Dirichlet BC local DOF {} out of range (max {})",
                    bc.local_dof,
                    dofs_per_node - 1
                )));
            }
            let global_dof = bc.node_id * dofs_per_node + bc.local_dof;
            constrained_dofs.insert(global_dof, bc.value);
        }

        // Process Neumann BCs
        for bc in &bc_set.neumann {
            if bc.node_id >= mesh.num_nodes() {
                return Err(FemError::BoundaryError(format!(
                    "Neumann BC node {} out of range (mesh has {} nodes)",
                    bc.node_id,
                    mesh.num_nodes()
                )));
            }
            if bc.local_dof >= dofs_per_node {
                return Err(FemError::BoundaryError(format!(
                    "Neumann BC local DOF {} out of range (max {})",
                    bc.local_dof,
                    dofs_per_node - 1
                )));
            }
            let global_dof = bc.node_id * dofs_per_node + bc.local_dof;
            *neumann_loads.entry(global_dof).or_insert(0.0) += bc.value;
        }

        Ok(Self {
            constrained_dofs,
            neumann_loads,
            penalty_factor: bc_set.penalty_factor,
        })
    }
}

/// Applies boundary conditions to the global system
///
/// This function modifies the stiffness matrix and load vector
/// by applying:
/// 1. Neumann BCs (added to load vector)
/// 2. Dirichlet BCs (either by elimination or penalty method)
///
/// # Arguments
/// * `system` - The assembled global system
/// * `bc` - The processed boundary conditions
///
/// # Returns
/// Modified stiffness matrix and load vector with boundary conditions applied
pub fn apply_boundary_conditions(
    system: &GlobalSystem,
    bc: &BoundaryCondition,
) -> FemResult<(CsrMatrix, Vec<f64>)> {
    let n = system.stiffness.n_rows;
    let mut load = system.load.clone();

    // Apply Neumann BCs (add to load vector)
    for (&dof, &force) in &bc.neumann_loads {
        if dof < n {
            load[dof] += force;
        }
    }

    // Apply Dirichlet BCs
    if let Some(penalty) = bc.penalty_factor {
        // Penalty method
        apply_dirichlet_penalty(system, &bc.constrained_dofs, &mut load, penalty)
    } else {
        // Elimination method (more accurate)
        apply_dirichlet_elimination(system, &bc.constrained_dofs, &load)
    }
}

/// Applies Dirichlet BCs using the elimination (row/column zeroing) method
///
/// For each constrained DOF i with prescribed value g_i:
/// 1. Set K[i][j] = 0 for all j != i, K[i][i] = 1
/// 2. Set K[j][i] = 0 for all j != i
/// 3. Subtract K[j][i] * g_i from f[j] for all free j
/// 4. Set f[i] = g_i
fn apply_dirichlet_elimination(
    system: &GlobalSystem,
    constrained: &HashMap<usize, f64>,
    load: &[f64],
) -> FemResult<(CsrMatrix, Vec<f64>)> {
    let n = system.stiffness.n_rows;
    let mut f = load.to_vec();

    // Modify load vector: f_j -= K_ji * g_i for constrained i
    for (&constrained_dof, &prescribed_val) in constrained {
        for j in 0..n {
            if !constrained.contains_key(&j) {
                let k_ji = system.stiffness.get(j, constrained_dof);
                f[j] -= k_ji * prescribed_val;
            }
        }
    }

    // Build modified stiffness matrix
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

    for i in 0..n {
        if constrained.contains_key(&i) {
            // Constrained row: only diagonal = 1
            triplets.push((i, i, 1.0));
            f[i] = constrained.get(&i).copied().unwrap_or(0.0);
        } else {
            // Free row: copy entries, but zero out constrained columns
            let start = system.stiffness.row_ptr[i];
            let end = system.stiffness.row_ptr[i + 1];
            for idx in start..end {
                let j = system.stiffness.col_idx[idx];
                if !constrained.contains_key(&j) {
                    triplets.push((i, j, system.stiffness.values[idx]));
                }
            }
        }
    }

    let k_mod = CsrMatrix::from_triplets(n, n, &triplets);
    Ok((k_mod, f))
}

/// Applies Dirichlet BCs using the penalty method
///
/// For each constrained DOF i with prescribed value g_i:
/// K[i][i] += penalty
/// f[i] += penalty * g_i
fn apply_dirichlet_penalty(
    system: &GlobalSystem,
    constrained: &HashMap<usize, f64>,
    load: &mut [f64],
    penalty: f64,
) -> FemResult<(CsrMatrix, Vec<f64>)> {
    // Copy the stiffness matrix and modify diagonal
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

    // Copy all existing entries
    for i in 0..system.stiffness.n_rows {
        let start = system.stiffness.row_ptr[i];
        let end = system.stiffness.row_ptr[i + 1];
        for idx in start..end {
            triplets.push((
                i,
                system.stiffness.col_idx[idx],
                system.stiffness.values[idx],
            ));
        }
    }

    // Add penalty terms
    for (&dof, &value) in constrained {
        triplets.push((dof, dof, penalty));
        load[dof] += penalty * value;
    }

    let k_mod =
        CsrMatrix::from_triplets(system.stiffness.n_rows, system.stiffness.n_cols, &triplets);

    Ok((k_mod, load.to_vec()))
}

/// Applies natural boundary conditions (zero-flux)
///
/// Natural boundary conditions are automatically satisfied by the FEM
/// formulation and don't require explicit modification. This function
/// is provided for completeness and documentation.
pub fn apply_natural_bc(_system: &GlobalSystem) -> FemResult<()> {
    // Natural BCs are automatically satisfied - no modification needed
    Ok(())
}

/// Validates that boundary conditions are consistent with the mesh
pub fn validate_boundary_conditions(
    bc_set: &BoundaryConditionSet,
    mesh: &Mesh,
    dofs_per_node: usize,
) -> FemResult<()> {
    // Check Dirichlet BCs
    for bc in &bc_set.dirichlet {
        if bc.node_id >= mesh.num_nodes() {
            return Err(FemError::BoundaryError(format!(
                "Dirichlet BC references non-existent node {}",
                bc.node_id
            )));
        }
        if bc.local_dof >= dofs_per_node {
            return Err(FemError::BoundaryError(format!(
                "Dirichlet BC references invalid DOF {} (max {})",
                bc.local_dof,
                dofs_per_node - 1
            )));
        }
    }

    // Check Neumann BCs
    for bc in &bc_set.neumann {
        if bc.node_id >= mesh.num_nodes() {
            return Err(FemError::BoundaryError(format!(
                "Neumann BC references non-existent node {}",
                bc.node_id
            )));
        }
        if bc.local_dof >= dofs_per_node {
            return Err(FemError::BoundaryError(format!(
                "Neumann BC references invalid DOF {} (max {})",
                bc.local_dof,
                dofs_per_node - 1
            )));
        }
    }

    // Check for conflicting BCs (same DOF has both Dirichlet and Neumann)
    let dirichlet_dofs: std::collections::HashSet<(usize, usize)> = bc_set
        .dirichlet
        .iter()
        .map(|bc| (bc.node_id, bc.local_dof))
        .collect();

    for bc in &bc_set.neumann {
        if dirichlet_dofs.contains(&(bc.node_id, bc.local_dof)) {
            return Err(FemError::BoundaryError(format!(
                "Conflicting BCs: node {} DOF {} has both Dirichlet and Neumann conditions",
                bc.node_id, bc.local_dof
            )));
        }
    }

    // Check penalty factor if set
    if let Some(penalty) = bc_set.penalty_factor {
        if penalty <= 0.0 {
            return Err(FemError::BoundaryError(
                "Penalty factor must be positive".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::assembly::assemble_global_system;
    use super::super::mesh::Mesh;
    use super::*;

    #[test]
    fn test_dirichlet_bc_creation() {
        let bc = DirichletBc::scalar(5, 100.0);
        assert_eq!(bc.node_id, 5);
        assert_eq!(bc.local_dof, 0);
        assert!((bc.value - 100.0).abs() < 1e-12);

        let bc2 = DirichletBc::new(3, 1, -2.5);
        assert_eq!(bc2.node_id, 3);
        assert_eq!(bc2.local_dof, 1);
        assert!((bc2.value - (-2.5)).abs() < 1e-12);
    }

    #[test]
    fn test_neumann_bc_creation() {
        let bc = NeumannBc::new(3, 1, -50.0);
        assert_eq!(bc.node_id, 3);
        assert_eq!(bc.local_dof, 1);
        assert!((bc.value - (-50.0)).abs() < 1e-12);

        let bc2 = NeumannBc::scalar(0, 10.0);
        assert_eq!(bc2.node_id, 0);
        assert_eq!(bc2.local_dof, 0);
    }

    #[test]
    fn test_bc_set_builder() {
        let mut bc_set = BoundaryConditionSet::new();
        assert!(bc_set.is_empty());

        bc_set
            .add_scalar_dirichlet(&[(0, 0.0), (10, 1.0)])
            .add_point_force(5, 0, 100.0);

        assert_eq!(bc_set.num_dirichlet(), 2);
        assert_eq!(bc_set.num_neumann(), 1);
        assert!(!bc_set.is_empty());
    }

    #[test]
    fn test_bc_processing() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 5).expect("mesh gen should succeed");
        let mut bc_set = BoundaryConditionSet::new();
        bc_set.add_scalar_dirichlet(&[(0, 0.0), (5, 1.0)]);

        let bc =
            BoundaryCondition::from_set(&bc_set, &mesh, 1).expect("BC processing should succeed");

        assert_eq!(bc.constrained_dofs.len(), 2);
        assert!((bc.constrained_dofs[&0] - 0.0).abs() < 1e-12);
        assert!((bc.constrained_dofs[&5] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_bc_processing_invalid_node() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 5).expect("mesh gen should succeed");
        let mut bc_set = BoundaryConditionSet::new();
        bc_set.add_scalar_dirichlet(&[(100, 0.0)]); // Invalid node

        let result = BoundaryCondition::from_set(&bc_set, &mesh, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_dirichlet_elimination() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 2).expect("mesh gen should succeed");
        let system =
            assemble_global_system(&mesh, 1.0, &|_| 0.0, 2, 1).expect("assembly should succeed");

        let mut bc_set = BoundaryConditionSet::new();
        bc_set.add_scalar_dirichlet(&[(0, 0.0), (2, 1.0)]);

        let bc =
            BoundaryCondition::from_set(&bc_set, &mesh, 1).expect("BC processing should succeed");

        let (k_mod, f_mod) =
            apply_boundary_conditions(&system, &bc).expect("BC application should succeed");

        // Constrained rows should have diagonal = 1, off-diagonal = 0
        assert!((k_mod.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((k_mod.get(2, 2) - 1.0).abs() < 1e-12);
        assert!(k_mod.get(0, 1).abs() < 1e-12);

        // Load vector at constrained DOFs = prescribed value
        assert!((f_mod[0] - 0.0).abs() < 1e-12);
        assert!((f_mod[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_neumann_bc() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 4).expect("mesh gen should succeed");
        let system =
            assemble_global_system(&mesh, 1.0, &|_| 0.0, 2, 1).expect("assembly should succeed");

        let mut bc_set = BoundaryConditionSet::new();
        bc_set
            .add_dirichlet(DirichletBc::scalar(0, 0.0))
            .add_neumann(NeumannBc::scalar(4, 10.0));

        let bc =
            BoundaryCondition::from_set(&bc_set, &mesh, 1).expect("BC processing should succeed");

        let (_k_mod, f_mod) =
            apply_boundary_conditions(&system, &bc).expect("BC application should succeed");

        // Neumann load at node 4 (DOF 4) should include the force
        assert!((f_mod[4] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_penalty_method() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 2).expect("mesh gen should succeed");
        let system =
            assemble_global_system(&mesh, 1.0, &|_| 0.0, 2, 1).expect("assembly should succeed");

        let mut bc_set = BoundaryConditionSet::new();
        bc_set
            .add_scalar_dirichlet(&[(0, 0.0), (2, 1.0)])
            .set_penalty_factor(1e10);

        let bc =
            BoundaryCondition::from_set(&bc_set, &mesh, 1).expect("BC processing should succeed");

        let (k_mod, f_mod) =
            apply_boundary_conditions(&system, &bc).expect("BC application should succeed");

        // Diagonal should be augmented by penalty
        let k00 = k_mod.get(0, 0);
        assert!(k00 > 1e9); // Should have penalty added

        // Load at constrained DOF 2 should have penalty * value
        assert!(f_mod[2] > 1e9);
    }

    #[test]
    fn test_validate_boundary_conditions() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 5).expect("mesh gen should succeed");

        // Valid BCs
        let mut bc_set = BoundaryConditionSet::new();
        bc_set.add_scalar_dirichlet(&[(0, 0.0), (5, 1.0)]);
        assert!(validate_boundary_conditions(&bc_set, &mesh, 1).is_ok());

        // Invalid: node out of range
        let mut bad_bc = BoundaryConditionSet::new();
        bad_bc.add_scalar_dirichlet(&[(100, 0.0)]);
        assert!(validate_boundary_conditions(&bad_bc, &mesh, 1).is_err());

        // Invalid: conflicting BCs
        let mut conflict = BoundaryConditionSet::new();
        conflict
            .add_dirichlet(DirichletBc::scalar(0, 0.0))
            .add_neumann(NeumannBc::scalar(0, 1.0));
        assert!(validate_boundary_conditions(&conflict, &mesh, 1).is_err());
    }

    #[test]
    fn test_validate_penalty_factor() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 3).expect("mesh gen should succeed");

        let mut bc_set = BoundaryConditionSet::new();
        bc_set.add_scalar_dirichlet(&[(0, 0.0)]);
        bc_set.set_penalty_factor(-1.0);
        assert!(validate_boundary_conditions(&bc_set, &mesh, 1).is_err());
    }
}
