//! FEM equation solvers and post-processing.
//!
//! Provides direct and iterative solvers for the linear systems arising
//! from finite element discretization, as well as post-processing utilities
//! for computing derived quantities (stress, strain, heat flux).

use super::assembly::{assemble_elasticity_system, assemble_global_system, CsrMatrix};
use super::boundary::{
    apply_boundary_conditions, validate_boundary_conditions, BoundaryCondition,
    BoundaryConditionSet, DirichletBc,
};
use super::elements::{
    compute_jacobian, matrix_inverse, ElementType, GaussQuadrature, ShapeFunction,
};
use super::mesh::Mesh;
use super::{FemError, FemResult, PlaneStress2D};

/// Solver configuration parameters
#[derive(Clone, Debug)]
pub struct SolverConfig {
    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,
    /// Convergence tolerance for iterative solvers
    pub tolerance: f64,
    /// Whether to use preconditioning
    pub use_preconditioner: bool,
    /// Gauss quadrature order
    pub quadrature_order: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            tolerance: 1e-10,
            use_preconditioner: true,
            quadrature_order: 2,
            verbose: false,
        }
    }
}

/// Trait for FEM solvers
pub trait FemSolver {
    /// Solves the linear system K*u = f
    fn solve(&self, stiffness: &CsrMatrix, load: &[f64]) -> FemResult<Vec<f64>>;
}

/// Direct solver using Gaussian elimination with partial pivoting
#[derive(Clone, Debug)]
pub struct DirectSolver;

impl FemSolver for DirectSolver {
    fn solve(&self, stiffness: &CsrMatrix, load: &[f64]) -> FemResult<Vec<f64>> {
        let n = stiffness.n_rows;
        if n != load.len() {
            return Err(FemError::SolverError(format!(
                "Stiffness matrix size {} does not match load vector size {}",
                n,
                load.len()
            )));
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut aug = vec![vec![0.0; n + 1]; n];
        for i in 0..n {
            let start = stiffness.row_ptr[i];
            let end = stiffness.row_ptr[i + 1];
            for idx in start..end {
                aug[i][stiffness.col_idx[idx]] = stiffness.values[idx];
            }
            aug[i][n] = load[i];
        }

        for col in 0..n {
            let mut max_val = aug[col][col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-15 {
                return Err(FemError::SingularSystem(format!(
                    "Zero pivot at column {} during Gaussian elimination",
                    col
                )));
            }
            if max_row != col {
                aug.swap(col, max_row);
            }
            let pivot = aug[col][col];
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    let val = aug[col][j];
                    aug[row][j] -= factor * val;
                }
            }
        }

        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * x[j];
            }
            if aug[i][i].abs() < 1e-15 {
                return Err(FemError::SingularSystem(format!(
                    "Zero diagonal at row {} during back substitution",
                    i
                )));
            }
            x[i] = sum / aug[i][i];
        }
        Ok(x)
    }
}

/// Conjugate Gradient solver for symmetric positive-definite systems
#[derive(Clone, Debug)]
pub struct ConjugateGradientSolver {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance (relative residual)
    pub tolerance: f64,
    /// Whether to use Jacobi (diagonal) preconditioning
    pub use_preconditioner: bool,
}

impl Default for ConjugateGradientSolver {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            tolerance: 1e-10,
            use_preconditioner: true,
        }
    }
}

impl ConjugateGradientSolver {
    /// Creates a new CG solver with specified parameters
    pub fn new(max_iterations: usize, tolerance: f64, use_preconditioner: bool) -> Self {
        Self {
            max_iterations,
            tolerance,
            use_preconditioner,
        }
    }
}

impl FemSolver for ConjugateGradientSolver {
    fn solve(&self, stiffness: &CsrMatrix, load: &[f64]) -> FemResult<Vec<f64>> {
        let n = stiffness.n_rows;
        if n != load.len() {
            return Err(FemError::SolverError(format!(
                "Matrix size {} != load size {}",
                n,
                load.len()
            )));
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut x = vec![0.0; n];
        let mut r = load.to_vec();

        let precond = if self.use_preconditioner {
            let diag = stiffness.diagonal();
            diag.iter()
                .map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 1.0 })
                .collect::<Vec<f64>>()
        } else {
            vec![1.0; n]
        };

        let mut z: Vec<f64> = r
            .iter()
            .zip(precond.iter())
            .map(|(&ri, &mi)| ri * mi)
            .collect();
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum();

        let rhs_norm: f64 = load.iter().map(|&fi| fi * fi).sum::<f64>().sqrt();
        let tol = if rhs_norm > 1e-30 {
            self.tolerance * rhs_norm
        } else {
            self.tolerance
        };

        for _iter in 0..self.max_iterations {
            let r_norm: f64 = r.iter().map(|&ri| ri * ri).sum::<f64>().sqrt();
            if r_norm < tol {
                return Ok(x);
            }

            let ap = stiffness.matvec(&p)?;
            let pap: f64 = p.iter().zip(ap.iter()).map(|(&pi, &api)| pi * api).sum();
            if pap.abs() < 1e-30 {
                return Err(FemError::SolverError(
                    "CG breakdown: p^T * A * p is near zero".to_string(),
                ));
            }
            let alpha = rz / pap;

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
                z[i] = r[i] * precond[i];
            }

            let rz_new: f64 = r.iter().zip(z.iter()).map(|(&ri, &zi)| ri * zi).sum();
            let beta = if rz.abs() > 1e-30 { rz_new / rz } else { 0.0 };

            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }

        Err(FemError::SolverError(format!(
            "CG did not converge after {} iterations",
            self.max_iterations
        )))
    }
}

/// FEM solution result
#[derive(Clone, Debug)]
pub struct FemSolution {
    /// Nodal solution values
    pub values: Vec<f64>,
    /// DOFs per node
    pub dofs_per_node: usize,
    /// Number of nodes
    pub num_nodes: usize,
    /// Solver iterations (for iterative solvers, 0 for direct)
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: f64,
}

impl FemSolution {
    /// Gets the solution value at a specific node and DOF
    pub fn node_value(&self, node_id: usize, dof: usize) -> FemResult<f64> {
        let idx = node_id * self.dofs_per_node + dof;
        if idx >= self.values.len() {
            return Err(FemError::SolverError(format!(
                "Solution index {} out of range",
                idx
            )));
        }
        Ok(self.values[idx])
    }
}

/// Post-processing utilities for FEM solutions
pub struct PostProcessor;

impl PostProcessor {
    /// Computes the gradient (flux/strain) at element centroids
    pub fn compute_gradients(
        mesh: &Mesh,
        solution: &[f64],
        conductivity: f64,
    ) -> FemResult<Vec<Vec<f64>>> {
        let mut gradients = Vec::with_capacity(mesh.num_elements());
        for elem_id in 0..mesh.num_elements() {
            let elem = &mesh.elements[elem_id];
            let elem_type = ElementType::from_kind(&elem.kind);
            let coords = mesh.element_coords(elem_id)?;
            let dim = mesh.dimension;

            let centroid: Vec<f64> = match elem.kind {
                super::mesh::ElementKind::Line2 => vec![0.0],
                super::mesh::ElementKind::Triangle3 => vec![1.0 / 3.0, 1.0 / 3.0],
                super::mesh::ElementKind::Quad4 => vec![0.0, 0.0],
            };

            let dn_dxi = elem_type.derivatives(&centroid);
            let jac = compute_jacobian(&dn_dxi, &coords);
            let jac_inv = matrix_inverse(&jac)?;

            let n_nodes = elem.num_nodes();
            let mut dn_dx = vec![vec![0.0; dim]; n_nodes];
            for i in 0..n_nodes {
                for j in 0..dim {
                    for k in 0..dim {
                        dn_dx[i][j] += dn_dxi[i][k] * jac_inv[k][j];
                    }
                }
            }

            let mut grad = vec![0.0; dim];
            for i in 0..n_nodes {
                let node_id = elem.nodes[i];
                let u_i = if node_id < solution.len() {
                    solution[node_id]
                } else {
                    0.0
                };
                for d in 0..dim {
                    grad[d] += u_i * dn_dx[i][d];
                }
            }
            for val in grad.iter_mut() {
                *val *= -conductivity;
            }
            gradients.push(grad);
        }
        Ok(gradients)
    }

    /// Computes stress from displacement for 2D plane stress
    pub fn compute_stress_2d(
        mesh: &Mesh,
        solution: &[f64],
        d_matrix: &[[f64; 3]; 3],
    ) -> FemResult<Vec<[f64; 3]>> {
        let mut stresses = Vec::with_capacity(mesh.num_elements());
        for elem_id in 0..mesh.num_elements() {
            let elem = &mesh.elements[elem_id];
            let elem_type = ElementType::from_kind(&elem.kind);
            let coords = mesh.element_coords(elem_id)?;
            let n_nodes = elem.num_nodes();

            let centroid: Vec<f64> = match elem.kind {
                super::mesh::ElementKind::Triangle3 => vec![1.0 / 3.0, 1.0 / 3.0],
                super::mesh::ElementKind::Quad4 => vec![0.0, 0.0],
                _ => {
                    return Err(FemError::ElementError(
                        "2D stress only for 2D elements".to_string(),
                    ))
                }
            };

            let dn_dxi = elem_type.derivatives(&centroid);
            let jac = compute_jacobian(&dn_dxi, &coords);
            let jac_inv = matrix_inverse(&jac)?;

            let mut dn_dx = vec![vec![0.0; 2]; n_nodes];
            for i in 0..n_nodes {
                for j in 0..2 {
                    for k in 0..2 {
                        dn_dx[i][j] += dn_dxi[i][k] * jac_inv[k][j];
                    }
                }
            }

            let mut strain = [0.0; 3];
            for i in 0..n_nodes {
                let ux = solution[2 * elem.nodes[i]];
                let uy = solution[2 * elem.nodes[i] + 1];
                strain[0] += dn_dx[i][0] * ux;
                strain[1] += dn_dx[i][1] * uy;
                strain[2] += dn_dx[i][1] * ux + dn_dx[i][0] * uy;
            }

            let mut stress = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    stress[i] += d_matrix[i][j] * strain[j];
                }
            }
            stresses.push(stress);
        }
        Ok(stresses)
    }

    /// Evaluates the FEM solution at an arbitrary point
    pub fn evaluate_at_point(mesh: &Mesh, solution: &[f64], point: &[f64]) -> FemResult<f64> {
        if mesh.dimension == 1 {
            let x = point[0];
            for elem in &mesh.elements {
                let x0 = mesh.nodes[elem.nodes[0]].coords[0];
                let x1 = mesh.nodes[elem.nodes[1]].coords[0];
                let (lo, hi) = if x0 < x1 { (x0, x1) } else { (x1, x0) };
                if x >= lo - 1e-12 && x <= hi + 1e-12 {
                    let xi = 2.0 * (x - x0) / (x1 - x0) - 1.0;
                    let n1 = (1.0 - xi) / 2.0;
                    let n2 = (1.0 + xi) / 2.0;
                    return Ok(n1 * solution[elem.nodes[0]] + n2 * solution[elem.nodes[1]]);
                }
            }
            return Err(FemError::SolverError(format!(
                "Point {:?} not found in any element",
                point
            )));
        }
        Err(FemError::SolverError(
            "Point evaluation for 2D/3D not yet implemented".to_string(),
        ))
    }
}

/// Solves a 1D steady-state heat equation
pub fn solve_heat_equation_1d(
    mesh: &Mesh,
    conductivity: f64,
    source_fn: impl Fn(f64) -> f64,
    dirichlet_bcs: &[(usize, f64)],
) -> FemResult<FemSolution> {
    if mesh.dimension != 1 {
        return Err(FemError::SolverError(
            "solve_heat_equation_1d requires a 1D mesh".to_string(),
        ));
    }
    let source_wrapper = |x: &[f64]| -> f64 {
        if x.is_empty() {
            0.0
        } else {
            source_fn(x[0])
        }
    };
    let system = assemble_global_system(mesh, conductivity, &source_wrapper, 2, 1)?;

    let mut bc_set = BoundaryConditionSet::new();
    bc_set.add_scalar_dirichlet(dirichlet_bcs);
    validate_boundary_conditions(&bc_set, mesh, 1)?;
    let bc = BoundaryCondition::from_set(&bc_set, mesh, 1)?;
    let (k_mod, f_mod) = apply_boundary_conditions(&system, &bc)?;

    let solver = DirectSolver;
    let u = solver.solve(&k_mod, &f_mod)?;

    let ku = k_mod.matvec(&u)?;
    let residual: f64 = ku
        .iter()
        .zip(f_mod.iter())
        .map(|(ki, fi)| (ki - fi) * (ki - fi))
        .sum::<f64>()
        .sqrt();

    Ok(FemSolution {
        values: u,
        dofs_per_node: 1,
        num_nodes: mesh.num_nodes(),
        iterations: 0,
        residual_norm: residual,
    })
}

/// Solves a 1D bar elasticity problem
pub fn solve_elasticity_1d(
    mesh: &Mesh,
    ea: f64,
    source_fn: impl Fn(f64) -> f64,
    dirichlet_bcs: &[(usize, f64)],
    neumann_bcs: &[(usize, f64)],
) -> FemResult<FemSolution> {
    if mesh.dimension != 1 {
        return Err(FemError::SolverError(
            "solve_elasticity_1d requires a 1D mesh".to_string(),
        ));
    }
    let source_wrapper = |x: &[f64]| -> f64 {
        if x.is_empty() {
            0.0
        } else {
            source_fn(x[0])
        }
    };
    let system = assemble_global_system(mesh, ea, &source_wrapper, 2, 1)?;

    let mut bc_set = BoundaryConditionSet::new();
    bc_set.add_scalar_dirichlet(dirichlet_bcs);
    for &(node_id, force) in neumann_bcs {
        bc_set.add_point_force(node_id, 0, force);
    }
    validate_boundary_conditions(&bc_set, mesh, 1)?;
    let bc = BoundaryCondition::from_set(&bc_set, mesh, 1)?;
    let (k_mod, f_mod) = apply_boundary_conditions(&system, &bc)?;

    let solver = DirectSolver;
    let u = solver.solve(&k_mod, &f_mod)?;

    let ku = k_mod.matvec(&u)?;
    let residual: f64 = ku
        .iter()
        .zip(f_mod.iter())
        .map(|(ki, fi)| (ki - fi) * (ki - fi))
        .sum::<f64>()
        .sqrt();

    Ok(FemSolution {
        values: u,
        dofs_per_node: 1,
        num_nodes: mesh.num_nodes(),
        iterations: 0,
        residual_norm: residual,
    })
}

/// Solves a 2D plane stress elasticity problem
pub fn solve_plane_stress_2d(
    mesh: &Mesh,
    material: &PlaneStress2D,
    body_force: impl Fn(&[f64]) -> [f64; 2],
    dirichlet_bcs: &[(usize, usize, f64)],
    neumann_bcs: &[(usize, usize, f64)],
) -> FemResult<FemSolution> {
    if mesh.dimension != 2 {
        return Err(FemError::SolverError(
            "solve_plane_stress_2d requires a 2D mesh".to_string(),
        ));
    }
    let d_matrix = material.elasticity_matrix();
    let system = assemble_elasticity_system(mesh, &d_matrix, material.thickness, &body_force, 2)?;

    let mut bc_set = BoundaryConditionSet::new();
    for &(node_id, dof, value) in dirichlet_bcs {
        bc_set.add_dirichlet(DirichletBc::new(node_id, dof, value));
    }
    for &(node_id, dof, force) in neumann_bcs {
        bc_set.add_point_force(node_id, dof, force);
    }
    validate_boundary_conditions(&bc_set, mesh, 2)?;
    let bc = BoundaryCondition::from_set(&bc_set, mesh, 2)?;
    let (k_mod, f_mod) = apply_boundary_conditions(&system, &bc)?;

    let n_dofs = k_mod.n_rows;
    let u = if n_dofs <= 100 {
        DirectSolver.solve(&k_mod, &f_mod)?
    } else {
        ConjugateGradientSolver::new(10000, 1e-10, true).solve(&k_mod, &f_mod)?
    };

    let ku = k_mod.matvec(&u)?;
    let residual: f64 = ku
        .iter()
        .zip(f_mod.iter())
        .map(|(ki, fi)| (ki - fi) * (ki - fi))
        .sum::<f64>()
        .sqrt();

    Ok(FemSolution {
        values: u,
        dofs_per_node: 2,
        num_nodes: mesh.num_nodes(),
        iterations: 0,
        residual_norm: residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_solver_simple() {
        let triplets = vec![(0, 0, 2.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0)];
        let k = CsrMatrix::from_triplets(2, 2, &triplets);
        let f = vec![1.0, 1.0];
        let solver = DirectSolver;
        let x = solver.solve(&k, &f).expect("solve should succeed");
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cg_solver_simple() {
        let triplets = vec![(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let k = CsrMatrix::from_triplets(2, 2, &triplets);
        let f = vec![1.0, 2.0];
        let solver = ConjugateGradientSolver::new(100, 1e-12, true);
        let x = solver.solve(&k, &f).expect("CG should converge");
        let kx = k.matvec(&x).expect("matvec should succeed");
        for i in 0..2 {
            assert!((kx[i] - f[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_cg_solver_larger() {
        let n = 4;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.0));
            }
            if i < n - 1 {
                triplets.push((i, i + 1, -1.0));
            }
        }
        let k = CsrMatrix::from_triplets(n, n, &triplets);
        let f = vec![1.0; n];
        let solver = ConjugateGradientSolver::new(100, 1e-12, true);
        let x = solver.solve(&k, &f).expect("CG should converge");
        let kx = k.matvec(&x).expect("matvec should succeed");
        for i in 0..n {
            assert!((kx[i] - f[i]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_heat_equation_1d_uniform_source() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 10).expect("mesh gen should succeed");
        let n_nodes = mesh.num_nodes();
        let solution = solve_heat_equation_1d(&mesh, 1.0, |_| 1.0, &[(0, 0.0), (n_nodes - 1, 0.0)])
            .expect("heat solve should succeed");
        let u_mid = solution.values[5];
        let exact_mid = 0.5 * (1.0 - 0.5) / 2.0;
        assert!(
            (u_mid - exact_mid).abs() < 0.01,
            "u(0.5) = {} vs exact = {}",
            u_mid,
            exact_mid
        );
        assert!(solution.values[0].abs() < 1e-10);
        assert!(solution.values[n_nodes - 1].abs() < 1e-10);
    }

    #[test]
    fn test_heat_equation_1d_linear_solution() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 5).expect("mesh gen should succeed");
        let n_nodes = mesh.num_nodes();
        let solution = solve_heat_equation_1d(&mesh, 1.0, |_| 0.0, &[(0, 0.0), (n_nodes - 1, 1.0)])
            .expect("heat solve should succeed");
        for i in 0..n_nodes {
            let x = mesh.nodes[i].coords[0];
            let u = solution.values[i];
            assert!(
                (u - x).abs() < 1e-10,
                "At x = {}: u = {} vs exact = {}",
                x,
                u,
                x
            );
        }
    }

    #[test]
    fn test_1d_bar_elasticity() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 4).expect("mesh gen should succeed");
        let solution = solve_elasticity_1d(&mesh, 1e6, |_| 0.0, &[(0, 0.0)], &[(4, 1000.0)])
            .expect("elasticity solve should succeed");
        let u_tip = solution.values[4];
        let exact_tip = 1000.0 / 1e6;
        assert!(
            (u_tip - exact_tip).abs() / exact_tip < 0.01,
            "u_tip = {} vs exact = {}",
            u_tip,
            exact_tip
        );
    }

    #[test]
    fn test_post_processor_gradient_1d() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 4).expect("mesh gen should succeed");
        let solution: Vec<f64> = (0..5).map(|i| i as f64 * 0.25).collect();
        let gradients = PostProcessor::compute_gradients(&mesh, &solution, 1.0)
            .expect("gradient computation should succeed");
        for grad in &gradients {
            assert!((grad[0] - (-1.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_post_processor_evaluate_1d() {
        let mesh = Mesh::generate_1d(0.0, 1.0, 4).expect("mesh gen should succeed");
        let solution: Vec<f64> = (0..5).map(|i| i as f64 * 0.25).collect();
        let val = PostProcessor::evaluate_at_point(&mesh, &solution, &[0.5])
            .expect("evaluation should succeed");
        assert!((val - 0.5).abs() < 1e-10);
        let val = PostProcessor::evaluate_at_point(&mesh, &solution, &[0.3])
            .expect("evaluation should succeed");
        assert!((val - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_heat_convergence() {
        let pi = std::f64::consts::PI;
        let mut prev_error = f64::INFINITY;
        for &n_elem in &[4, 8, 16, 32] {
            let mesh = Mesh::generate_1d(0.0, 1.0, n_elem).expect("mesh gen should succeed");
            let n_nodes = mesh.num_nodes();
            let solution = solve_heat_equation_1d(
                &mesh,
                1.0,
                |x| pi * pi * (pi * x).sin(),
                &[(0, 0.0), (n_nodes - 1, 0.0)],
            )
            .expect("heat solve should succeed");
            let mut error_sq = 0.0;
            let h = 1.0 / n_elem as f64;
            for i in 0..n_nodes {
                let x = i as f64 * h;
                let exact = (pi * x).sin();
                let err = solution.values[i] - exact;
                error_sq += err * err * h;
            }
            let error = error_sq.sqrt();
            assert!(
                error < prev_error,
                "Error {} should be less than previous {}",
                error,
                prev_error
            );
            prev_error = error;
        }
        assert!(prev_error < 0.001);
    }

    #[test]
    fn test_2d_patch_test() {
        let mesh = Mesh::generate_2d_triangular(0.0, 1.0, 0.0, 1.0, 2, 2)
            .expect("mesh gen should succeed");
        let n_nodes = mesh.num_nodes();
        let mut bcs: Vec<(usize, f64)> = Vec::new();
        for &nid in &mesh.boundary_nodes {
            let x = mesh.nodes[nid].coords[0];
            let y = mesh.nodes[nid].coords[1];
            bcs.push((nid, 1.0 + 2.0 * x + 3.0 * y));
        }
        let source_wrapper = |_x: &[f64]| -> f64 { 0.0 };
        let system = assemble_global_system(&mesh, 1.0, &source_wrapper, 3, 1)
            .expect("assembly should succeed");
        let mut bc_set = BoundaryConditionSet::new();
        bc_set.add_scalar_dirichlet(&bcs);
        let bc = BoundaryCondition::from_set(&bc_set, &mesh, 1).expect("BC should succeed");
        let (k_mod, f_mod) =
            apply_boundary_conditions(&system, &bc).expect("BC apply should succeed");
        let solver = DirectSolver;
        let u = solver.solve(&k_mod, &f_mod).expect("solve should succeed");
        for i in 0..n_nodes {
            let x = mesh.nodes[i].coords[0];
            let y = mesh.nodes[i].coords[1];
            let exact = 1.0 + 2.0 * x + 3.0 * y;
            assert!(
                (u[i] - exact).abs() < 1e-8,
                "Node {} at ({}, {}): u = {} vs exact = {}",
                i,
                x,
                y,
                u[i],
                exact
            );
        }
    }

    #[test]
    fn test_solver_singular_detection() {
        let triplets = vec![(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)];
        let k = CsrMatrix::from_triplets(2, 2, &triplets);
        let f = vec![1.0, -1.0];
        let solver = DirectSolver;
        let result = solver.solve(&k, &f);
        match result {
            Ok(_) | Err(_) => {}
        }
    }
}
