//! Finite Element Method (FEM) Module for NumRS2
//!
//! This module provides a comprehensive Finite Element Method implementation
//! for solving partial differential equations (PDEs) on arbitrary domains.
//! It supports 1D and 2D problems including heat conduction, linear elasticity,
//! and general elliptic PDEs.
//!
//! # Module Organization
//!
//! - [`mesh`]: Mesh data structures (nodes, elements, connectivity) and mesh generation
//! - [`elements`]: Element types, shape functions, Jacobian computation, and quadrature
//! - [`assembly`]: Global stiffness matrix and load vector assembly
//! - [`solvers`]: Direct and iterative solvers with post-processing capabilities
//! - [`boundary`]: Boundary condition handling (Dirichlet, Neumann, penalty method)
//!
//! # Theory
//!
//! The Finite Element Method approximates a weak-form PDE by discretizing the domain
//! into elements and expanding the solution in terms of piecewise polynomial basis
//! (shape) functions. For a general elliptic PDE:
//!
//! ```text
//! -div(k * grad(u)) + c * u = f   in Omega
//!                          u = g   on Gamma_D  (Dirichlet)
//!              k * du/dn = h       on Gamma_N  (Neumann)
//! ```
//!
//! The weak form leads to the system: **K u = F**
//!
//! where **K** is the global stiffness matrix and **F** is the global load vector.
//!
//! # Example
//!
//! ```rust,ignore
//! use numrs2::new_modules::fem::*;
//!
//! // Create a 1D mesh with 10 elements on [0, 1]
//! let mesh = Mesh::generate_1d(0.0, 1.0, 10)?;
//!
//! // Solve the heat equation -u'' = 1 with u(0) = 0, u(1) = 0
//! let result = solve_heat_equation_1d(&mesh, 1.0, |_x| 1.0, &[(0, 0.0), (10, 0.0)])?;
//! ```
//!
//! # SCIRS2 Policy Compliance
//!
//! This module uses `scirs2_core::ndarray` for all array operations and
//! follows all NumRS2/SCIRS2 ecosystem policies.

pub mod assembly;
pub mod boundary;
pub mod elements;
pub mod mesh;
pub mod solvers;

// Re-exports for convenience
pub use assembly::{assemble_global_system, DofNumbering};
pub use boundary::{
    apply_boundary_conditions, BoundaryCondition, BoundaryConditionSet, DirichletBc, NeumannBc,
};
pub use elements::{
    ElementStiffness, ElementType, GaussQuadrature, LineElement, QuadElement, ShapeFunction,
    TriElement,
};
pub use mesh::{Element, Mesh, MeshQuality, Node};
pub use solvers::{
    solve_elasticity_1d, solve_heat_equation_1d, ConjugateGradientSolver, DirectSolver, FemSolver,
    PostProcessor, SolverConfig,
};

use crate::error::NumRs2Error;
use std::fmt;
use thiserror::Error;

/// FEM-specific error types
#[derive(Error, Debug, Clone)]
pub enum FemError {
    /// Mesh-related error
    #[error("Mesh error: {0}")]
    MeshError(String),

    /// Element-related error
    #[error("Element error: {0}")]
    ElementError(String),

    /// Assembly error
    #[error("Assembly error: {0}")]
    AssemblyError(String),

    /// Solver convergence error
    #[error("Solver error: {0}")]
    SolverError(String),

    /// Boundary condition error
    #[error("Boundary condition error: {0}")]
    BoundaryError(String),

    /// Invalid input parameters
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Singular or ill-conditioned system
    #[error("Singular system: {0}")]
    SingularSystem(String),

    /// Numerical precision issue
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Result type for FEM operations
pub type FemResult<T> = std::result::Result<T, FemError>;

impl From<FemError> for NumRs2Error {
    fn from(err: FemError) -> Self {
        NumRs2Error::ComputationError(format!("FEM: {}", err))
    }
}

impl From<NumRs2Error> for FemError {
    fn from(err: NumRs2Error) -> Self {
        FemError::NumericalError(format!("NumRS2: {}", err))
    }
}

/// Trait for defining a FEM problem
///
/// Implementors provide the material properties, source terms,
/// and boundary conditions for a specific PDE.
pub trait FemProblem {
    /// Returns the spatial dimension of the problem (1, 2, or 3)
    fn dimension(&self) -> usize;

    /// Evaluates the material conductivity/stiffness at a point
    ///
    /// For heat equation: thermal conductivity k
    /// For elasticity: Young's modulus E
    fn conductivity(&self, point: &[f64]) -> f64;

    /// Evaluates the source term at a point
    ///
    /// For heat equation: heat source f
    /// For elasticity: body force
    fn source_term(&self, point: &[f64]) -> f64;

    /// Returns the number of degrees of freedom per node
    fn dofs_per_node(&self) -> usize {
        1
    }

    /// Returns a descriptive name for the problem
    fn name(&self) -> &str;
}

/// 1D heat conduction problem definition
///
/// Solves: -d/dx(k * du/dx) = f(x)  on [a, b]
pub struct HeatConduction1D {
    /// Thermal conductivity (constant)
    pub conductivity_val: f64,
    /// Source function (as boxed closure)
    source_fn: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Problem name
    problem_name: String,
}

impl HeatConduction1D {
    /// Creates a new 1D heat conduction problem
    ///
    /// # Arguments
    /// * `conductivity` - Thermal conductivity (positive constant)
    /// * `source` - Source term function f(x)
    pub fn new(
        conductivity: f64,
        source: impl Fn(f64) -> f64 + Send + Sync + 'static,
    ) -> FemResult<Self> {
        if conductivity <= 0.0 {
            return Err(FemError::InvalidParameter(
                "Conductivity must be positive".to_string(),
            ));
        }
        Ok(Self {
            conductivity_val: conductivity,
            source_fn: Box::new(source),
            problem_name: "1D Heat Conduction".to_string(),
        })
    }

    /// Evaluates the source function
    pub fn eval_source(&self, x: f64) -> f64 {
        (self.source_fn)(x)
    }
}

impl FemProblem for HeatConduction1D {
    fn dimension(&self) -> usize {
        1
    }

    fn conductivity(&self, _point: &[f64]) -> f64 {
        self.conductivity_val
    }

    fn source_term(&self, point: &[f64]) -> f64 {
        if point.is_empty() {
            return 0.0;
        }
        (self.source_fn)(point[0])
    }

    fn name(&self) -> &str {
        &self.problem_name
    }
}

impl fmt::Debug for HeatConduction1D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeatConduction1D")
            .field("conductivity", &self.conductivity_val)
            .field("name", &self.problem_name)
            .finish()
    }
}

/// 1D bar elasticity problem definition
///
/// Solves: -d/dx(E*A * du/dx) = f(x)  on [0, L]
pub struct BarElasticity1D {
    /// Young's modulus
    pub youngs_modulus: f64,
    /// Cross-sectional area
    pub cross_section_area: f64,
    /// Source function (body force per unit length)
    source_fn: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Problem name
    problem_name: String,
}

impl BarElasticity1D {
    /// Creates a new 1D bar elasticity problem
    ///
    /// # Arguments
    /// * `youngs_modulus` - Young's modulus E (positive)
    /// * `cross_section_area` - Cross-sectional area A (positive)
    /// * `source` - Body force per unit length f(x)
    pub fn new(
        youngs_modulus: f64,
        cross_section_area: f64,
        source: impl Fn(f64) -> f64 + Send + Sync + 'static,
    ) -> FemResult<Self> {
        if youngs_modulus <= 0.0 {
            return Err(FemError::InvalidParameter(
                "Young's modulus must be positive".to_string(),
            ));
        }
        if cross_section_area <= 0.0 {
            return Err(FemError::InvalidParameter(
                "Cross-section area must be positive".to_string(),
            ));
        }
        Ok(Self {
            youngs_modulus,
            cross_section_area,
            source_fn: Box::new(source),
            problem_name: "1D Bar Elasticity".to_string(),
        })
    }

    /// Returns the axial stiffness EA
    pub fn axial_stiffness(&self) -> f64 {
        self.youngs_modulus * self.cross_section_area
    }

    /// Evaluates the body force
    pub fn eval_source(&self, x: f64) -> f64 {
        (self.source_fn)(x)
    }
}

impl FemProblem for BarElasticity1D {
    fn dimension(&self) -> usize {
        1
    }

    fn conductivity(&self, _point: &[f64]) -> f64 {
        self.youngs_modulus * self.cross_section_area
    }

    fn source_term(&self, point: &[f64]) -> f64 {
        if point.is_empty() {
            return 0.0;
        }
        (self.source_fn)(point[0])
    }

    fn name(&self) -> &str {
        &self.problem_name
    }
}

impl fmt::Debug for BarElasticity1D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BarElasticity1D")
            .field("youngs_modulus", &self.youngs_modulus)
            .field("cross_section_area", &self.cross_section_area)
            .field("name", &self.problem_name)
            .finish()
    }
}

/// 2D plane stress elasticity problem definition
///
/// Solves the 2D linear elasticity equations under plane stress assumption.
pub struct PlaneStress2D {
    /// Young's modulus
    pub youngs_modulus: f64,
    /// Poisson's ratio
    pub poisson_ratio: f64,
    /// Thickness
    pub thickness: f64,
    /// Problem name
    problem_name: String,
}

impl PlaneStress2D {
    /// Creates a new 2D plane stress problem
    ///
    /// # Arguments
    /// * `youngs_modulus` - Young's modulus E (positive)
    /// * `poisson_ratio` - Poisson's ratio nu (between -1 and 0.5)
    /// * `thickness` - Plate thickness t (positive)
    pub fn new(youngs_modulus: f64, poisson_ratio: f64, thickness: f64) -> FemResult<Self> {
        if youngs_modulus <= 0.0 {
            return Err(FemError::InvalidParameter(
                "Young's modulus must be positive".to_string(),
            ));
        }
        if poisson_ratio <= -1.0 || poisson_ratio >= 0.5 {
            return Err(FemError::InvalidParameter(
                "Poisson's ratio must be in (-1, 0.5)".to_string(),
            ));
        }
        if thickness <= 0.0 {
            return Err(FemError::InvalidParameter(
                "Thickness must be positive".to_string(),
            ));
        }
        Ok(Self {
            youngs_modulus,
            poisson_ratio,
            thickness,
            problem_name: "2D Plane Stress".to_string(),
        })
    }

    /// Returns the elasticity matrix D for plane stress
    ///
    /// ```text
    ///        E        [ 1    nu   0           ]
    /// D = --------- * [ nu   1    0           ]
    ///     1 - nu^2    [ 0    0    (1-nu)/2    ]
    /// ```
    pub fn elasticity_matrix(&self) -> [[f64; 3]; 3] {
        let e = self.youngs_modulus;
        let nu = self.poisson_ratio;
        let factor = e / (1.0 - nu * nu);
        [
            [factor, factor * nu, 0.0],
            [factor * nu, factor, 0.0],
            [0.0, 0.0, factor * (1.0 - nu) / 2.0],
        ]
    }
}

impl FemProblem for PlaneStress2D {
    fn dimension(&self) -> usize {
        2
    }

    fn conductivity(&self, _point: &[f64]) -> f64 {
        self.youngs_modulus
    }

    fn source_term(&self, _point: &[f64]) -> f64 {
        0.0
    }

    fn dofs_per_node(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        &self.problem_name
    }
}

impl fmt::Debug for PlaneStress2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlaneStress2D")
            .field("youngs_modulus", &self.youngs_modulus)
            .field("poisson_ratio", &self.poisson_ratio)
            .field("thickness", &self.thickness)
            .field("name", &self.problem_name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fem_error_creation() {
        let err = FemError::MeshError("bad mesh".to_string());
        assert!(format!("{}", err).contains("bad mesh"));
    }

    #[test]
    fn test_fem_error_conversion() {
        let fem_err = FemError::SolverError("did not converge".to_string());
        let numrs_err: NumRs2Error = fem_err.into();
        assert!(format!("{}", numrs_err).contains("FEM"));
    }

    #[test]
    fn test_heat_conduction_1d() {
        let problem = HeatConduction1D::new(1.0, |x| x * x).expect("valid problem");
        assert_eq!(problem.dimension(), 1);
        assert_eq!(problem.conductivity(&[0.5]), 1.0);
        assert!((problem.source_term(&[2.0]) - 4.0).abs() < 1e-12);
        assert_eq!(problem.dofs_per_node(), 1);
    }

    #[test]
    fn test_heat_conduction_invalid() {
        let result = HeatConduction1D::new(-1.0, |_| 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_bar_elasticity_1d() {
        let problem = BarElasticity1D::new(200e9, 0.01, |_| 1000.0).expect("valid problem");
        assert_eq!(problem.dimension(), 1);
        assert!((problem.axial_stiffness() - 2e9).abs() < 1.0);
        assert!((problem.source_term(&[0.5]) - 1000.0).abs() < 1e-12);
    }

    #[test]
    fn test_plane_stress_2d() {
        let problem = PlaneStress2D::new(200e9, 0.3, 0.01).expect("valid problem");
        assert_eq!(problem.dimension(), 2);
        assert_eq!(problem.dofs_per_node(), 2);

        let d = problem.elasticity_matrix();
        // Check symmetry: D[0][1] == D[1][0]
        assert!((d[0][1] - d[1][0]).abs() < 1e-6);
        // Check D[2][2] = E*(1-nu)/(2*(1-nu^2))
        let expected_d33 = 200e9 * (1.0 - 0.3) / (2.0 * (1.0 - 0.3 * 0.3));
        assert!((d[2][2] - expected_d33).abs() / expected_d33 < 1e-10);
    }

    #[test]
    fn test_plane_stress_invalid_poisson() {
        let result = PlaneStress2D::new(200e9, 0.5, 0.01);
        assert!(result.is_err());
    }
}
