//! Global stiffness matrix and load vector assembly.
//!
//! This module handles the assembly of element-level contributions into
//! the global system of equations. It manages DOF numbering, sparse
//! matrix storage in CSR format, and the element-to-global mapping.
//!
//! # Assembly Process
//!
//! 1. Number the global degrees of freedom (DOFs)
//! 2. For each element:
//!    a. Compute the element stiffness matrix K_e and load vector f_e
//!    b. Map local DOF indices to global DOF indices
//!    c. Add K_e and f_e contributions to the global system
//! 3. Apply boundary conditions
//!
//! # Sparse Storage
//!
//! The global stiffness matrix is stored in Compressed Sparse Row (CSR)
//! format for memory efficiency, since FEM stiffness matrices are typically
//! very sparse.

use super::elements::compute_element_stiffness;
use super::mesh::Mesh;
use super::{FemError, FemResult};
use std::collections::HashMap;

/// Result of sparse assembly: (stiffness matrix, load vector, DOF mapping, Dirichlet BCs)
type SparseAssemblyResult = (CsrMatrix, Vec<f64>, Vec<usize>, Vec<(usize, f64)>);

/// DOF numbering scheme for the global system
///
/// Maps nodes to their global degrees of freedom indices.
#[derive(Clone, Debug)]
pub struct DofNumbering {
    /// Maps (node_id, local_dof) -> global_dof_index
    pub node_dof_map: HashMap<(usize, usize), usize>,
    /// Total number of DOFs
    pub total_dofs: usize,
    /// Number of DOFs per node
    pub dofs_per_node: usize,
    /// Maps global_dof_index -> (node_id, local_dof)
    pub inverse_map: Vec<(usize, usize)>,
}

impl DofNumbering {
    /// Creates a DOF numbering for a mesh
    ///
    /// # Arguments
    /// * `mesh` - The finite element mesh
    /// * `dofs_per_node` - Number of DOFs per node (1 for scalar, 2 for 2D elasticity)
    pub fn new(mesh: &Mesh, dofs_per_node: usize) -> FemResult<Self> {
        if dofs_per_node == 0 {
            return Err(FemError::AssemblyError(
                "DOFs per node must be positive".to_string(),
            ));
        }

        let n_nodes = mesh.num_nodes();
        let total_dofs = n_nodes * dofs_per_node;

        let mut node_dof_map = HashMap::with_capacity(total_dofs);
        let mut inverse_map = Vec::with_capacity(total_dofs);

        for node_id in 0..n_nodes {
            for local_dof in 0..dofs_per_node {
                let global_dof = node_id * dofs_per_node + local_dof;
                node_dof_map.insert((node_id, local_dof), global_dof);
                inverse_map.push((node_id, local_dof));
            }
        }

        Ok(Self {
            node_dof_map,
            total_dofs,
            dofs_per_node,
            inverse_map,
        })
    }

    /// Gets the global DOF index for a node and local DOF
    pub fn global_dof(&self, node_id: usize, local_dof: usize) -> FemResult<usize> {
        self.node_dof_map
            .get(&(node_id, local_dof))
            .copied()
            .ok_or_else(|| {
                FemError::AssemblyError(format!(
                    "No global DOF for node {} local DOF {}",
                    node_id, local_dof
                ))
            })
    }

    /// Gets the global DOF indices for all DOFs of an element
    pub fn element_dofs(&self, mesh: &Mesh, element_id: usize) -> FemResult<Vec<usize>> {
        if element_id >= mesh.elements.len() {
            return Err(FemError::AssemblyError(format!(
                "Element {} out of range",
                element_id
            )));
        }

        let elem = &mesh.elements[element_id];
        let mut global_dofs = Vec::with_capacity(elem.nodes.len() * self.dofs_per_node);

        for &node_id in &elem.nodes {
            for local_dof in 0..self.dofs_per_node {
                global_dofs.push(self.global_dof(node_id, local_dof)?);
            }
        }

        Ok(global_dofs)
    }

    /// Gets the (node_id, local_dof) pair for a global DOF index
    pub fn node_and_dof(&self, global_dof: usize) -> FemResult<(usize, usize)> {
        if global_dof >= self.inverse_map.len() {
            return Err(FemError::AssemblyError(format!(
                "Global DOF {} out of range (total_dofs = {})",
                global_dof, self.total_dofs
            )));
        }
        Ok(self.inverse_map[global_dof])
    }
}

/// Sparse matrix in Compressed Sparse Row (CSR) format
///
/// Stores only non-zero entries for memory efficiency.
/// The matrix entry A\[i\]\[j\] is stored if it is structurally non-zero
/// (i.e., if there exists an element connecting node i and node j).
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Row pointers: row_ptr\[i\] is the index in col_idx/values where row i starts
    pub row_ptr: Vec<usize>,
    /// Column indices for non-zero entries
    pub col_idx: Vec<usize>,
    /// Values of non-zero entries
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Creates a new empty CSR matrix
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            row_ptr: vec![0; n_rows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Creates a CSR matrix from a dense matrix stored as `Vec<Vec<f64>>`
    ///
    /// Only stores entries with absolute value > tolerance.
    pub fn from_dense(dense: &[Vec<f64>], tolerance: f64) -> Self {
        let n_rows = dense.len();
        let n_cols = if n_rows > 0 { dense[0].len() } else { 0 };

        let mut row_ptr = vec![0usize; n_rows + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for (i, row) in dense.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val.abs() > tolerance {
                    col_idx.push(j);
                    values.push(val);
                }
            }
            row_ptr[i + 1] = col_idx.len();
        }

        Self {
            n_rows,
            n_cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Creates a CSR matrix from triplets (row, col, value)
    ///
    /// Duplicate entries are summed.
    pub fn from_triplets(n_rows: usize, n_cols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        // Accumulate values in a HashMap
        let mut entries: HashMap<(usize, usize), f64> = HashMap::new();
        for &(row, col, val) in triplets {
            *entries.entry((row, col)).or_insert(0.0) += val;
        }

        // Sort by (row, col)
        let mut sorted_entries: Vec<((usize, usize), f64)> = entries.into_iter().collect();
        sorted_entries.sort_by_key(|&((r, c), _)| (r, c));

        let mut row_ptr = vec![0usize; n_rows + 1];
        let mut col_idx = Vec::with_capacity(sorted_entries.len());
        let mut values = Vec::with_capacity(sorted_entries.len());

        let mut current_row = 0;
        for ((row, col), val) in &sorted_entries {
            while current_row < *row {
                current_row += 1;
                row_ptr[current_row] = col_idx.len();
            }
            col_idx.push(*col);
            values.push(*val);
        }
        while current_row < n_rows {
            current_row += 1;
            row_ptr[current_row] = col_idx.len();
        }

        Self {
            n_rows,
            n_cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Gets the value at position (row, col)
    ///
    /// Returns 0.0 if the position is not stored (structurally zero).
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.n_rows || col >= self.n_cols {
            return 0.0;
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];

        for idx in start..end {
            if self.col_idx[idx] == col {
                return self.values[idx];
            }
        }
        0.0
    }

    /// Returns the number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Performs sparse matrix-vector multiplication: y = A * x
    pub fn matvec(&self, x: &[f64]) -> FemResult<Vec<f64>> {
        if x.len() != self.n_cols {
            return Err(FemError::AssemblyError(format!(
                "Vector size {} does not match matrix columns {}",
                x.len(),
                self.n_cols
            )));
        }

        let mut y = vec![0.0; self.n_rows];

        for i in 0..self.n_rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                y[i] += self.values[idx] * x[self.col_idx[idx]];
            }
        }

        Ok(y)
    }

    /// Converts to a dense matrix (for small systems and debugging)
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.n_cols]; self.n_rows];
        for i in 0..self.n_rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                dense[i][self.col_idx[idx]] = self.values[idx];
            }
        }
        dense
    }

    /// Converts to a flat row-major vector (for solver interface)
    pub fn to_flat(&self) -> Vec<f64> {
        let mut flat = vec![0.0; self.n_rows * self.n_cols];
        for i in 0..self.n_rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                flat[i * self.n_cols + self.col_idx[idx]] = self.values[idx];
            }
        }
        flat
    }

    /// Checks if the matrix is symmetric (within tolerance)
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        if self.n_rows != self.n_cols {
            return false;
        }
        for i in 0..self.n_rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                let j = self.col_idx[idx];
                let val_ij = self.values[idx];
                let val_ji = self.get(j, i);
                if (val_ij - val_ji).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Returns the diagonal entries
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.n_rows.min(self.n_cols);
        let mut diag = vec![0.0; n];
        for i in 0..n {
            diag[i] = self.get(i, i);
        }
        diag
    }
}

/// Result of global system assembly
#[derive(Clone, Debug)]
pub struct GlobalSystem {
    /// Global stiffness matrix in CSR format
    pub stiffness: CsrMatrix,
    /// Global load (force) vector
    pub load: Vec<f64>,
    /// DOF numbering used
    pub dof_numbering: DofNumbering,
}

/// Assembles the global stiffness matrix and load vector
///
/// This is the main assembly routine that:
/// 1. Numbers the DOFs
/// 2. Computes element stiffness matrices
/// 3. Assembles into the global sparse system
///
/// # Arguments
/// * `mesh` - The finite element mesh
/// * `conductivity` - Material conductivity/stiffness (uniform)
/// * `source_fn` - Source term function
/// * `quadrature_order` - Gauss quadrature order
/// * `dofs_per_node` - Number of DOFs per node
///
/// # Returns
/// The assembled global system (stiffness matrix + load vector)
pub fn assemble_global_system(
    mesh: &Mesh,
    conductivity: f64,
    source_fn: &dyn Fn(&[f64]) -> f64,
    quadrature_order: usize,
    dofs_per_node: usize,
) -> FemResult<GlobalSystem> {
    let dof_numbering = DofNumbering::new(mesh, dofs_per_node)?;
    let n_dofs = dof_numbering.total_dofs;

    // Collect triplets (row, col, value) for sparse assembly
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut load = vec![0.0; n_dofs];

    for elem_id in 0..mesh.num_elements() {
        let (ke, fe) =
            compute_element_stiffness(mesh, elem_id, conductivity, source_fn, quadrature_order)?;

        let global_dofs = dof_numbering.element_dofs(mesh, elem_id)?;

        // Assemble element stiffness into global triplets
        for (i, &gi) in global_dofs.iter().enumerate() {
            for (j, &gj) in global_dofs.iter().enumerate() {
                if i < ke.len() && j < ke[i].len() && ke[i][j].abs() > 1e-30 {
                    triplets.push((gi, gj, ke[i][j]));
                }
            }
            // Assemble load vector
            if i < fe.len() {
                load[gi] += fe[i];
            }
        }
    }

    let stiffness = CsrMatrix::from_triplets(n_dofs, n_dofs, &triplets);

    Ok(GlobalSystem {
        stiffness,
        load,
        dof_numbering,
    })
}

/// Assembles the global system for 2D plane stress elasticity
///
/// Each node has 2 DOFs (u_x, u_y), so the global system size is 2 * n_nodes.
///
/// # Arguments
/// * `mesh` - The 2D finite element mesh
/// * `d_matrix` - 3x3 constitutive matrix
/// * `thickness` - Element thickness
/// * `body_force` - Body force function returning [fx, fy]
/// * `quadrature_order` - Gauss quadrature order
pub fn assemble_elasticity_system(
    mesh: &Mesh,
    d_matrix: &[[f64; 3]; 3],
    thickness: f64,
    body_force: &dyn Fn(&[f64]) -> [f64; 2],
    quadrature_order: usize,
) -> FemResult<GlobalSystem> {
    use super::elements::{
        compute_element_stiffness_2d_elasticity, compute_jacobian, matrix_determinant, ElementType,
        GaussQuadrature, ShapeFunction,
    };

    let dof_numbering = DofNumbering::new(mesh, 2)?;
    let n_dofs = dof_numbering.total_dofs;

    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    let mut load = vec![0.0; n_dofs];

    for elem_id in 0..mesh.num_elements() {
        let ke = compute_element_stiffness_2d_elasticity(
            mesh,
            elem_id,
            d_matrix,
            thickness,
            quadrature_order,
        )?;

        let global_dofs = dof_numbering.element_dofs(mesh, elem_id)?;
        let n_elem_dofs = global_dofs.len();

        // Assemble stiffness
        for i in 0..n_elem_dofs {
            for j in 0..n_elem_dofs {
                if i < ke.len() && j < ke[i].len() && ke[i][j].abs() > 1e-30 {
                    triplets.push((global_dofs[i], global_dofs[j], ke[i][j]));
                }
            }
        }

        // Assemble load vector from body forces
        let elem = &mesh.elements[elem_id];
        let coords = mesh.element_coords(elem_id)?;
        let elem_type = ElementType::from_kind(&elem.kind);
        let quadrature = GaussQuadrature::for_element(&elem.kind, quadrature_order)?;
        let n_nodes = elem.num_nodes();

        for q in 0..quadrature.num_points() {
            let xi = &quadrature.points[q];
            let w = quadrature.weights[q];

            let n_vals = elem_type.evaluate(xi);
            let dn_dxi = elem_type.derivatives(xi);
            let jac = compute_jacobian(&dn_dxi, &coords);
            let det_j = matrix_determinant(&jac)?;

            // Physical coordinates
            let mut phys = vec![0.0; 2];
            for d in 0..2 {
                for i in 0..n_nodes {
                    if i < coords.len() && d < coords[i].len() {
                        phys[d] += n_vals[i] * coords[i][d];
                    }
                }
            }

            let bf = body_force(&phys);
            let factor = det_j.abs() * thickness * w;

            for i in 0..n_nodes {
                let gi_x = global_dofs[2 * i];
                let gi_y = global_dofs[2 * i + 1];
                load[gi_x] += bf[0] * n_vals[i] * factor;
                load[gi_y] += bf[1] * n_vals[i] * factor;
            }
        }
    }

    let stiffness = CsrMatrix::from_triplets(n_dofs, n_dofs, &triplets);

    Ok(GlobalSystem {
        stiffness,
        load,
        dof_numbering,
    })
}

/// Condenses the global system by removing constrained DOFs
///
/// Creates a reduced system of equations by eliminating rows and columns
/// corresponding to prescribed (Dirichlet) DOFs.
///
/// # Arguments
/// * `system` - The full global system
/// * `constrained_dofs` - Map of constrained DOF index -> prescribed value
///
/// # Returns
/// (K_reduced, f_reduced, free_dofs, constrained_dofs_sorted)
pub fn condense_system(
    system: &GlobalSystem,
    constrained_dofs: &HashMap<usize, f64>,
) -> FemResult<SparseAssemblyResult> {
    let n = system.stiffness.n_rows;

    // Identify free (unconstrained) DOFs
    let free_dofs: Vec<usize> = (0..n)
        .filter(|dof| !constrained_dofs.contains_key(dof))
        .collect();
    let n_free = free_dofs.len();

    if n_free == 0 {
        return Err(FemError::AssemblyError(
            "All DOFs are constrained; no free DOFs remain".to_string(),
        ));
    }

    // Create mapping from old DOF index to new (reduced) index
    let mut old_to_new: HashMap<usize, usize> = HashMap::with_capacity(n_free);
    for (new_idx, &old_idx) in free_dofs.iter().enumerate() {
        old_to_new.insert(old_idx, new_idx);
    }

    // Build reduced load vector: f_free = f_free - K_fc * u_c
    let mut f_reduced = vec![0.0; n_free];
    for (new_i, &old_i) in free_dofs.iter().enumerate() {
        f_reduced[new_i] = system.load[old_i];

        // Subtract contributions from constrained DOFs
        let start = system.stiffness.row_ptr[old_i];
        let end = system.stiffness.row_ptr[old_i + 1];
        for idx in start..end {
            let old_j = system.stiffness.col_idx[idx];
            if let Some(&prescribed_val) = constrained_dofs.get(&old_j) {
                f_reduced[new_i] -= system.stiffness.values[idx] * prescribed_val;
            }
        }
    }

    // Build reduced stiffness matrix
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    for (new_i, &old_i) in free_dofs.iter().enumerate() {
        let start = system.stiffness.row_ptr[old_i];
        let end = system.stiffness.row_ptr[old_i + 1];
        for idx in start..end {
            let old_j = system.stiffness.col_idx[idx];
            if let Some(&new_j) = old_to_new.get(&old_j) {
                triplets.push((new_i, new_j, system.stiffness.values[idx]));
            }
        }
    }

    let k_reduced = CsrMatrix::from_triplets(n_free, n_free, &triplets);

    let mut constrained_sorted: Vec<(usize, f64)> =
        constrained_dofs.iter().map(|(&k, &v)| (k, v)).collect();
    constrained_sorted.sort_by_key(|&(k, _)| k);

    Ok((k_reduced, f_reduced, free_dofs, constrained_sorted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dof_numbering() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 4).expect("mesh gen should succeed");
        let dof = DofNumbering::new(&mesh, 1).expect("dof numbering should succeed");

        assert_eq!(dof.total_dofs, 5);
        assert_eq!(dof.global_dof(0, 0).expect("dof 0"), 0);
        assert_eq!(dof.global_dof(4, 0).expect("dof 4"), 4);
    }

    #[test]
    fn test_dof_numbering_2d() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 2).expect("mesh gen should succeed");
        let dof = DofNumbering::new(&mesh, 2).expect("dof numbering should succeed");

        assert_eq!(dof.total_dofs, 6); // 3 nodes * 2 DOFs
        assert_eq!(dof.global_dof(0, 0).expect("dof"), 0);
        assert_eq!(dof.global_dof(0, 1).expect("dof"), 1);
        assert_eq!(dof.global_dof(1, 0).expect("dof"), 2);
        assert_eq!(dof.global_dof(1, 1).expect("dof"), 3);
    }

    #[test]
    fn test_dof_numbering_invalid() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 2).expect("mesh gen should succeed");
        assert!(DofNumbering::new(&mesh, 0).is_err());

        let dof = DofNumbering::new(&mesh, 1).expect("dof numbering should succeed");
        assert!(dof.global_dof(10, 0).is_err()); // Node out of range
        assert!(dof.global_dof(0, 5).is_err()); // DOF out of range
    }

    #[test]
    fn test_dof_inverse_map() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 3).expect("mesh gen should succeed");
        let dof = DofNumbering::new(&mesh, 2).expect("dof numbering should succeed");

        let (node, local_dof) = dof.node_and_dof(5).expect("inverse map should succeed");
        assert_eq!(node, 2);
        assert_eq!(local_dof, 1);

        assert!(dof.node_and_dof(100).is_err());
    }

    #[test]
    fn test_csr_from_dense() {
        let dense = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 3.0, 0.0],
            vec![4.0, 0.0, 5.0],
        ];
        let csr = CsrMatrix::from_dense(&dense, 1e-15);

        assert_eq!(csr.n_rows, 3);
        assert_eq!(csr.n_cols, 3);
        assert_eq!(csr.nnz(), 5);
        assert!((csr.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((csr.get(0, 2) - 2.0).abs() < 1e-12);
        assert!((csr.get(1, 1) - 3.0).abs() < 1e-12);
        assert!((csr.get(2, 0) - 4.0).abs() < 1e-12);
        assert!(csr.get(0, 1).abs() < 1e-12); // Zero entry
    }

    #[test]
    fn test_csr_from_triplets() {
        let triplets = vec![(0, 0, 1.0), (0, 0, 2.0), (1, 1, 3.0), (0, 1, 4.0)];
        let csr = CsrMatrix::from_triplets(2, 2, &triplets);

        // (0,0) should be 1+2 = 3 (duplicates summed)
        assert!((csr.get(0, 0) - 3.0).abs() < 1e-12);
        assert!((csr.get(0, 1) - 4.0).abs() < 1e-12);
        assert!((csr.get(1, 1) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_csr_matvec() {
        let dense = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let x = vec![1.0, 2.0];
        let y = csr.matvec(&x).expect("matvec should succeed");
        assert!((y[0] - 5.0).abs() < 1e-12);
        assert!((y[1] - 11.0).abs() < 1e-12);
    }

    #[test]
    fn test_csr_symmetry() {
        let dense = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        assert!(csr.is_symmetric(1e-12));

        let asym = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let csr_asym = CsrMatrix::from_dense(&asym, 1e-15);
        assert!(!csr_asym.is_symmetric(1e-12));
    }

    #[test]
    fn test_csr_to_flat() {
        let dense = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let csr = CsrMatrix::from_dense(&dense, 1e-15);
        let flat = csr.to_flat();
        assert!((flat[0] - 1.0).abs() < 1e-12);
        assert!((flat[1] - 2.0).abs() < 1e-12);
        assert!((flat[2] - 3.0).abs() < 1e-12);
        assert!((flat[3] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_global_assembly_1d() {
        // 2-element mesh on [0, 1]
        let mesh = Mesh::generate_1d(0.0, 1.0, 2).expect("mesh gen should succeed");
        let system =
            assemble_global_system(&mesh, 1.0, &|_| 0.0, 2, 1).expect("assembly should succeed");

        // 3 nodes -> 3 DOFs
        assert_eq!(system.load.len(), 3);
        assert_eq!(system.stiffness.n_rows, 3);

        // Global stiffness should be symmetric
        assert!(system.stiffness.is_symmetric(1e-10));

        // For h=0.5, k=1: K_e = [[2, -2], [-2, 2]]
        // Global: [[2, -2, 0], [-2, 4, -2], [0, -2, 2]]
        let k00 = system.stiffness.get(0, 0);
        let k11 = system.stiffness.get(1, 1);
        let k01 = system.stiffness.get(0, 1);
        assert!((k00 - 2.0).abs() < 1e-10);
        assert!((k11 - 4.0).abs() < 1e-10); // 2+2 from two elements
        assert!((k01 - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_global_assembly_with_source() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 4).expect("mesh gen should succeed");
        let system =
            assemble_global_system(&mesh, 1.0, &|_| 1.0, 2, 1).expect("assembly should succeed");

        // Total load should equal integral of source over domain = 1*1 = 1
        let total_load: f64 = system.load.iter().sum();
        assert!((total_load - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_condense_system() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 2).expect("mesh gen should succeed");
        let system =
            assemble_global_system(&mesh, 1.0, &|_| 1.0, 2, 1).expect("assembly should succeed");

        // Constrain first and last nodes
        let mut constrained = HashMap::new();
        constrained.insert(0, 0.0);
        constrained.insert(2, 0.0);

        let (k_red, f_red, free_dofs, _) =
            condense_system(&system, &constrained).expect("condensation should succeed");

        assert_eq!(free_dofs.len(), 1);
        assert_eq!(free_dofs[0], 1);
        assert_eq!(k_red.n_rows, 1);
        assert_eq!(f_red.len(), 1);
    }

    #[test]
    fn test_assembly_symmetry_2d() {
        let mesh = Mesh::generate_2d_triangular(0.0, 1.0, 0.0, 1.0, 2, 2)
            .expect("mesh gen should succeed");
        let system =
            assemble_global_system(&mesh, 1.0, &|_| 0.0, 3, 1).expect("assembly should succeed");

        assert!(system.stiffness.is_symmetric(1e-10));
    }
}
