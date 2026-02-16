//! Finite element types, shape functions, and numerical integration.
//!
//! This module provides the fundamental building blocks for finite element
//! analysis: shape functions, their derivatives, Jacobian computation,
//! Gauss quadrature rules, and element stiffness matrix computation.
//!
//! # Supported Element Types
//!
//! - **Line2**: 2-node linear line element (1D)
//! - **Triangle3**: 3-node linear triangular element (2D)
//! - **Quad4**: 4-node bilinear quadrilateral element (2D)
//!
//! # Shape Functions
//!
//! Shape functions N_i(xi) satisfy the partition of unity and Kronecker delta
//! properties. For the reference element:
//!
//! - Line2: N_1(xi) = (1-xi)/2, N_2(xi) = (1+xi)/2, xi in [-1, 1]
//! - Triangle3: N_1 = 1-xi-eta, N_2 = xi, N_3 = eta, (xi,eta) in reference triangle
//! - Quad4: N_i = (1 +/- xi)(1 +/- eta)/4, (xi,eta) in [-1,1]^2

use super::mesh::{ElementKind, Mesh};
use super::{FemError, FemResult};

/// Trait for evaluating shape functions and their derivatives
pub trait ShapeFunction {
    /// Evaluates the shape functions at a point in the reference element
    ///
    /// # Arguments
    /// * `xi` - Natural coordinates in the reference element
    ///
    /// # Returns
    /// Vector of shape function values [N_1, N_2, ..., N_n]
    fn evaluate(&self, xi: &[f64]) -> Vec<f64>;

    /// Evaluates the derivatives of shape functions w.r.t. natural coordinates
    ///
    /// # Arguments
    /// * `xi` - Natural coordinates in the reference element
    ///
    /// # Returns
    /// Matrix of derivatives: dN_i/d(xi_j) stored as \[num_nodes\]\[num_dims\]
    fn derivatives(&self, xi: &[f64]) -> Vec<Vec<f64>>;

    /// Returns the number of nodes for this element type
    fn num_nodes(&self) -> usize;

    /// Returns the spatial dimension
    fn dimension(&self) -> usize;
}

/// 2-node linear line element on reference domain [-1, 1]
///
/// Shape functions:
/// - N_1(xi) = (1 - xi) / 2
/// - N_2(xi) = (1 + xi) / 2
#[derive(Clone, Debug)]
pub struct LineElement;

impl ShapeFunction for LineElement {
    fn evaluate(&self, xi: &[f64]) -> Vec<f64> {
        let x = if xi.is_empty() { 0.0 } else { xi[0] };
        vec![(1.0 - x) / 2.0, (1.0 + x) / 2.0]
    }

    fn derivatives(&self, _xi: &[f64]) -> Vec<Vec<f64>> {
        // dN/dxi is constant for linear elements
        vec![vec![-0.5], vec![0.5]]
    }

    fn num_nodes(&self) -> usize {
        2
    }

    fn dimension(&self) -> usize {
        1
    }
}

/// 3-node linear triangular element on reference triangle
///
/// Reference element: (0,0), (1,0), (0,1)
///
/// Shape functions:
/// - N_1(xi, eta) = 1 - xi - eta
/// - N_2(xi, eta) = xi
/// - N_3(xi, eta) = eta
#[derive(Clone, Debug)]
pub struct TriElement;

impl ShapeFunction for TriElement {
    fn evaluate(&self, xi: &[f64]) -> Vec<f64> {
        let (x, y) = if xi.len() >= 2 {
            (xi[0], xi[1])
        } else {
            (0.0, 0.0)
        };
        vec![1.0 - x - y, x, y]
    }

    fn derivatives(&self, _xi: &[f64]) -> Vec<Vec<f64>> {
        // Derivatives are constant for linear triangles
        vec![
            vec![-1.0, -1.0], // dN_1/dxi, dN_1/deta
            vec![1.0, 0.0],   // dN_2/dxi, dN_2/deta
            vec![0.0, 1.0],   // dN_3/dxi, dN_3/deta
        ]
    }

    fn num_nodes(&self) -> usize {
        3
    }

    fn dimension(&self) -> usize {
        2
    }
}

/// 4-node bilinear quadrilateral element on reference domain [-1,1]^2
///
/// Node ordering: (-1,-1), (1,-1), (1,1), (-1,1)
///
/// Shape functions:
/// - N_1(xi, eta) = (1-xi)(1-eta)/4
/// - N_2(xi, eta) = (1+xi)(1-eta)/4
/// - N_3(xi, eta) = (1+xi)(1+eta)/4
/// - N_4(xi, eta) = (1-xi)(1+eta)/4
#[derive(Clone, Debug)]
pub struct QuadElement;

impl ShapeFunction for QuadElement {
    fn evaluate(&self, xi: &[f64]) -> Vec<f64> {
        let (x, y) = if xi.len() >= 2 {
            (xi[0], xi[1])
        } else {
            (0.0, 0.0)
        };
        vec![
            (1.0 - x) * (1.0 - y) / 4.0,
            (1.0 + x) * (1.0 - y) / 4.0,
            (1.0 + x) * (1.0 + y) / 4.0,
            (1.0 - x) * (1.0 + y) / 4.0,
        ]
    }

    fn derivatives(&self, xi: &[f64]) -> Vec<Vec<f64>> {
        let (x, y) = if xi.len() >= 2 {
            (xi[0], xi[1])
        } else {
            (0.0, 0.0)
        };
        vec![
            vec![-(1.0 - y) / 4.0, -(1.0 - x) / 4.0],
            vec![(1.0 - y) / 4.0, -(1.0 + x) / 4.0],
            vec![(1.0 + y) / 4.0, (1.0 + x) / 4.0],
            vec![-(1.0 + y) / 4.0, (1.0 - x) / 4.0],
        ]
    }

    fn num_nodes(&self) -> usize {
        4
    }

    fn dimension(&self) -> usize {
        2
    }
}

/// Enum wrapper for element types
#[derive(Clone, Debug)]
pub enum ElementType {
    /// 1D line element
    Line(LineElement),
    /// 2D triangular element
    Triangle(TriElement),
    /// 2D quadrilateral element
    Quad(QuadElement),
}

impl ElementType {
    /// Creates the appropriate element type from an ElementKind
    pub fn from_kind(kind: &ElementKind) -> Self {
        match kind {
            ElementKind::Line2 => ElementType::Line(LineElement),
            ElementKind::Triangle3 => ElementType::Triangle(TriElement),
            ElementKind::Quad4 => ElementType::Quad(QuadElement),
        }
    }

    /// Returns the number of nodes for this element type
    pub fn node_count(&self) -> usize {
        match self {
            ElementType::Line(_) => 2,
            ElementType::Triangle(_) => 3,
            ElementType::Quad(_) => 4,
        }
    }

    /// Returns the spatial dimension of this element type
    pub fn spatial_dimension(&self) -> usize {
        match self {
            ElementType::Line(_) => 1,
            ElementType::Triangle(_) | ElementType::Quad(_) => 2,
        }
    }
}

impl ShapeFunction for ElementType {
    fn evaluate(&self, xi: &[f64]) -> Vec<f64> {
        match self {
            ElementType::Line(e) => e.evaluate(xi),
            ElementType::Triangle(e) => e.evaluate(xi),
            ElementType::Quad(e) => e.evaluate(xi),
        }
    }

    fn derivatives(&self, xi: &[f64]) -> Vec<Vec<f64>> {
        match self {
            ElementType::Line(e) => e.derivatives(xi),
            ElementType::Triangle(e) => e.derivatives(xi),
            ElementType::Quad(e) => e.derivatives(xi),
        }
    }

    fn num_nodes(&self) -> usize {
        match self {
            ElementType::Line(e) => e.num_nodes(),
            ElementType::Triangle(e) => e.num_nodes(),
            ElementType::Quad(e) => e.num_nodes(),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            ElementType::Line(e) => e.dimension(),
            ElementType::Triangle(e) => e.dimension(),
            ElementType::Quad(e) => e.dimension(),
        }
    }
}

/// Gauss quadrature rule
///
/// Provides integration points and weights for numerical integration
/// over reference elements.
#[derive(Clone, Debug)]
pub struct GaussQuadrature {
    /// Integration points in natural coordinates
    pub points: Vec<Vec<f64>>,
    /// Integration weights
    pub weights: Vec<f64>,
}

impl GaussQuadrature {
    /// Creates a Gauss quadrature rule for a 1D line element
    ///
    /// # Arguments
    /// * `order` - Number of quadrature points (1, 2, 3, 4, or 5)
    pub fn line(order: usize) -> FemResult<Self> {
        match order {
            1 => Ok(Self {
                points: vec![vec![0.0]],
                weights: vec![2.0],
            }),
            2 => {
                let p = 1.0 / 3.0_f64.sqrt();
                Ok(Self {
                    points: vec![vec![-p], vec![p]],
                    weights: vec![1.0, 1.0],
                })
            }
            3 => {
                let p = (3.0 / 5.0_f64).sqrt();
                Ok(Self {
                    points: vec![vec![-p], vec![0.0], vec![p]],
                    weights: vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
                })
            }
            4 => {
                let a = ((3.0 + 2.0 * (6.0 / 5.0_f64).sqrt()) / 7.0).sqrt();
                let b = ((3.0 - 2.0 * (6.0 / 5.0_f64).sqrt()) / 7.0).sqrt();
                let wa = (18.0 - 30.0_f64.sqrt()) / 36.0;
                let wb = (18.0 + 30.0_f64.sqrt()) / 36.0;
                Ok(Self {
                    points: vec![vec![-a], vec![-b], vec![b], vec![a]],
                    weights: vec![wa, wb, wb, wa],
                })
            }
            5 => {
                let p1 = (5.0 - 2.0 * (10.0 / 7.0_f64).sqrt()).sqrt() / 3.0;
                let p2 = (5.0 + 2.0 * (10.0 / 7.0_f64).sqrt()).sqrt() / 3.0;
                let w1 = (322.0 + 13.0 * 70.0_f64.sqrt()) / 900.0;
                let w2 = (322.0 - 13.0 * 70.0_f64.sqrt()) / 900.0;
                let w0 = 128.0 / 225.0;
                Ok(Self {
                    points: vec![vec![-p2], vec![-p1], vec![0.0], vec![p1], vec![p2]],
                    weights: vec![w2, w1, w0, w1, w2],
                })
            }
            _ => Err(FemError::ElementError(format!(
                "Unsupported quadrature order {} for line element (use 1-5)",
                order
            ))),
        }
    }

    /// Creates a Gauss quadrature rule for a triangular element
    ///
    /// # Arguments
    /// * `order` - Quadrature order (1, 3, or 4 points)
    pub fn triangle(order: usize) -> FemResult<Self> {
        match order {
            1 => Ok(Self {
                // Centroid rule
                points: vec![vec![1.0 / 3.0, 1.0 / 3.0]],
                weights: vec![0.5], // Area of reference triangle = 0.5
            }),
            3 => Ok(Self {
                // 3-point rule (exact for quadratic polynomials)
                points: vec![
                    vec![1.0 / 6.0, 1.0 / 6.0],
                    vec![2.0 / 3.0, 1.0 / 6.0],
                    vec![1.0 / 6.0, 2.0 / 3.0],
                ],
                weights: vec![1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
            }),
            4 => Ok(Self {
                // 4-point rule (degree of precision 3)
                points: vec![
                    vec![1.0 / 3.0, 1.0 / 3.0],
                    vec![1.0 / 5.0, 1.0 / 5.0],
                    vec![3.0 / 5.0, 1.0 / 5.0],
                    vec![1.0 / 5.0, 3.0 / 5.0],
                ],
                weights: vec![-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0],
            }),
            7 => {
                // 7-point rule for higher accuracy
                let a1 = 0.059_715_871_789_770;
                let b1 = 0.470_142_064_105_115;
                let a2 = 0.797_426_985_353_087;
                let b2 = 0.101_286_507_323_456;
                let w0 = 0.1125;
                let w1_val = 0.066_197_076_394_253;
                let w2_val = 0.062_969_590_272_414;
                Ok(Self {
                    points: vec![
                        vec![1.0 / 3.0, 1.0 / 3.0],
                        vec![a1, b1],
                        vec![b1, a1],
                        vec![b1, b1],
                        vec![a2, b2],
                        vec![b2, a2],
                        vec![b2, b2],
                    ],
                    weights: vec![w0, w1_val, w1_val, w1_val, w2_val, w2_val, w2_val],
                })
            }
            _ => Err(FemError::ElementError(format!(
                "Unsupported quadrature order {} for triangle (use 1, 3, 4, or 7)",
                order
            ))),
        }
    }

    /// Creates a Gauss quadrature rule for a quadrilateral element
    ///
    /// Uses tensor product of 1D Gauss rules.
    ///
    /// # Arguments
    /// * `order` - Number of points per direction (1, 2, 3, 4, or 5)
    pub fn quad(order: usize) -> FemResult<Self> {
        let line_rule = GaussQuadrature::line(order)?;

        let mut points = Vec::new();
        let mut weights = Vec::new();

        for (i, pi) in line_rule.points.iter().enumerate() {
            for (j, pj) in line_rule.points.iter().enumerate() {
                points.push(vec![pi[0], pj[0]]);
                weights.push(line_rule.weights[i] * line_rule.weights[j]);
            }
        }

        Ok(Self { points, weights })
    }

    /// Creates the appropriate quadrature rule for an element kind
    ///
    /// # Arguments
    /// * `kind` - Element type
    /// * `order` - Quadrature order
    pub fn for_element(kind: &ElementKind, order: usize) -> FemResult<Self> {
        match kind {
            ElementKind::Line2 => Self::line(order),
            ElementKind::Triangle3 => Self::triangle(order),
            ElementKind::Quad4 => Self::quad(order),
        }
    }

    /// Returns the number of integration points
    pub fn num_points(&self) -> usize {
        self.points.len()
    }
}

/// Computes the Jacobian matrix for an isoparametric element mapping
///
/// The Jacobian maps from reference coordinates (xi) to physical coordinates (x):
///
/// ```text
/// J_ij = dx_i / d(xi_j) = sum_k x_k * dN_k/d(xi_j)
/// ```
///
/// # Arguments
/// * `dshape` - Shape function derivatives w.r.t. natural coordinates \[num_nodes\]\[num_dims\]
/// * `coords` - Physical coordinates of element nodes \[num_nodes\]\[num_dims\]
///
/// # Returns
/// Jacobian matrix J\[dim\]\[dim\]
pub fn compute_jacobian(dshape: &[Vec<f64>], coords: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_nodes = dshape.len();
    let n_dim = if dshape.is_empty() {
        0
    } else {
        dshape[0].len()
    };

    let mut jacobian = vec![vec![0.0; n_dim]; n_dim];

    for i in 0..n_dim {
        for j in 0..n_dim {
            for k in 0..n_nodes {
                if k < coords.len() && i < coords[k].len() && j < dshape[k].len() {
                    jacobian[i][j] += coords[k][i] * dshape[k][j];
                }
            }
        }
    }

    jacobian
}

/// Computes the determinant of a square matrix (up to 3x3)
///
/// # Arguments
/// * `matrix` - Square matrix stored as `Vec<Vec<f64>>`
///
/// # Returns
/// The determinant value
pub fn matrix_determinant(matrix: &[Vec<f64>]) -> FemResult<f64> {
    let n = matrix.len();
    match n {
        0 => Ok(0.0),
        1 => Ok(matrix[0][0]),
        2 => Ok(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]),
        3 => Ok(
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]),
        ),
        _ => Err(FemError::ElementError(format!(
            "Determinant not implemented for {}x{} matrices",
            n, n
        ))),
    }
}

/// Computes the inverse of a square matrix (up to 3x3)
///
/// # Arguments
/// * `matrix` - Square matrix
///
/// # Returns
/// The inverse matrix, or an error if the matrix is singular
pub fn matrix_inverse(matrix: &[Vec<f64>]) -> FemResult<Vec<Vec<f64>>> {
    let n = matrix.len();
    let det = matrix_determinant(matrix)?;

    if det.abs() < 1e-30 {
        return Err(FemError::SingularSystem(
            "Jacobian matrix is singular or nearly singular".to_string(),
        ));
    }

    let inv_det = 1.0 / det;

    match n {
        1 => Ok(vec![vec![inv_det]]),
        2 => Ok(vec![
            vec![matrix[1][1] * inv_det, -matrix[0][1] * inv_det],
            vec![-matrix[1][0] * inv_det, matrix[0][0] * inv_det],
        ]),
        3 => {
            let mut inv = vec![vec![0.0; 3]; 3];
            inv[0][0] = (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det;
            inv[0][1] = (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det;
            inv[0][2] = (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det;
            inv[1][0] = (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det;
            inv[1][1] = (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det;
            inv[1][2] = (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det;
            inv[2][0] = (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det;
            inv[2][1] = (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det;
            inv[2][2] = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det;
            Ok(inv)
        }
        _ => Err(FemError::ElementError(format!(
            "Matrix inverse not implemented for {}x{} matrices",
            n, n
        ))),
    }
}

/// Trait for computing element stiffness matrices
pub trait ElementStiffness {
    /// Computes the element stiffness matrix and load vector
    ///
    /// # Arguments
    /// * `mesh` - The mesh containing the element
    /// * `element_id` - The element index
    /// * `conductivity` - Material conductivity/stiffness
    /// * `source_fn` - Source term evaluated at physical coordinates
    /// * `quadrature_order` - Number of quadrature points
    ///
    /// # Returns
    /// (stiffness_matrix, load_vector) for the element
    fn compute_stiffness(
        &self,
        mesh: &Mesh,
        element_id: usize,
        conductivity: f64,
        source_fn: &dyn Fn(&[f64]) -> f64,
        quadrature_order: usize,
    ) -> FemResult<(Vec<Vec<f64>>, Vec<f64>)>;
}

/// Computes the element stiffness matrix for a scalar PDE
///
/// For the equation -div(k * grad(u)) = f, the element stiffness is:
///
/// ```text
/// K_ij = integral( k * dN_i/dx * dN_j/dx * |J| * dxi )
/// ```
///
/// # Arguments
/// * `mesh` - The mesh
/// * `element_id` - Element index in the mesh
/// * `conductivity` - Material conductivity k
/// * `source_fn` - Source term function f(x)
/// * `quadrature_order` - Number of quadrature points
///
/// # Returns
/// (K_e, f_e): Element stiffness matrix and load vector
pub fn compute_element_stiffness(
    mesh: &Mesh,
    element_id: usize,
    conductivity: f64,
    source_fn: &dyn Fn(&[f64]) -> f64,
    quadrature_order: usize,
) -> FemResult<(Vec<Vec<f64>>, Vec<f64>)> {
    if element_id >= mesh.elements.len() {
        return Err(FemError::ElementError(format!(
            "Element index {} out of range",
            element_id
        )));
    }

    let elem = &mesh.elements[element_id];
    let elem_type = ElementType::from_kind(&elem.kind);
    let n_nodes = elem.num_nodes();
    let coords = mesh.element_coords(element_id)?;

    let quadrature = GaussQuadrature::for_element(&elem.kind, quadrature_order)?;

    let mut ke = vec![vec![0.0; n_nodes]; n_nodes];
    let mut fe = vec![0.0; n_nodes];

    for q in 0..quadrature.num_points() {
        let xi = &quadrature.points[q];
        let w = quadrature.weights[q];

        // Shape functions and derivatives
        let n_vals = elem_type.evaluate(xi);
        let dn_dxi = elem_type.derivatives(xi);

        // Jacobian
        let jac = compute_jacobian(&dn_dxi, &coords);
        let det_j = matrix_determinant(&jac)?;

        if det_j.abs() < 1e-30 {
            return Err(FemError::ElementError(format!(
                "Near-zero Jacobian determinant ({}) for element {}",
                det_j, element_id
            )));
        }

        let jac_inv = matrix_inverse(&jac)?;
        let dim = jac.len();

        // Transform derivatives to physical coordinates: dN/dx = dN/dxi * J^{-1}
        let mut dn_dx = vec![vec![0.0; dim]; n_nodes];
        for i in 0..n_nodes {
            for j in 0..dim {
                for k in 0..dim {
                    dn_dx[i][j] += dn_dxi[i][k] * jac_inv[k][j];
                }
            }
        }

        // Physical coordinates at this quadrature point
        let mut phys_coords = vec![0.0; dim];
        for d in 0..dim {
            for i in 0..n_nodes {
                phys_coords[d] += n_vals[i] * coords[i][d];
            }
        }

        let det_j_abs = det_j.abs();

        // Stiffness contribution: K_ij += k * (dN_i/dx . dN_j/dx) * |J| * w
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += dn_dx[i][d] * dn_dx[j][d];
                }
                ke[i][j] += conductivity * dot * det_j_abs * w;
            }
        }

        // Load vector contribution: f_i += f(x) * N_i * |J| * w
        let source_val = source_fn(&phys_coords);
        for i in 0..n_nodes {
            fe[i] += source_val * n_vals[i] * det_j_abs * w;
        }
    }

    Ok((ke, fe))
}

/// Computes the element stiffness matrix for 2D plane stress elasticity
///
/// For 2D elasticity, each node has 2 DOFs (u, v), so the element stiffness
/// matrix is (2*n_nodes) x (2*n_nodes).
///
/// # Arguments
/// * `mesh` - The mesh
/// * `element_id` - Element index
/// * `d_matrix` - 3x3 constitutive (elasticity) matrix
/// * `thickness` - Element thickness
/// * `quadrature_order` - Number of quadrature points per direction
///
/// # Returns
/// Element stiffness matrix for plane stress
pub fn compute_element_stiffness_2d_elasticity(
    mesh: &Mesh,
    element_id: usize,
    d_matrix: &[[f64; 3]; 3],
    thickness: f64,
    quadrature_order: usize,
) -> FemResult<Vec<Vec<f64>>> {
    if element_id >= mesh.elements.len() {
        return Err(FemError::ElementError(format!(
            "Element index {} out of range",
            element_id
        )));
    }

    let elem = &mesh.elements[element_id];
    let elem_type = ElementType::from_kind(&elem.kind);
    let n_nodes = elem.num_nodes();
    let n_dofs = 2 * n_nodes;
    let coords = mesh.element_coords(element_id)?;

    let quadrature = GaussQuadrature::for_element(&elem.kind, quadrature_order)?;

    let mut ke = vec![vec![0.0; n_dofs]; n_dofs];

    for q in 0..quadrature.num_points() {
        let xi = &quadrature.points[q];
        let w = quadrature.weights[q];

        let dn_dxi = elem_type.derivatives(xi);
        let jac = compute_jacobian(&dn_dxi, &coords);
        let det_j = matrix_determinant(&jac)?;

        if det_j.abs() < 1e-30 {
            return Err(FemError::ElementError(format!(
                "Near-zero Jacobian for element {}",
                element_id
            )));
        }

        let jac_inv = matrix_inverse(&jac)?;

        // Transform derivatives to physical space
        let mut dn_dx = vec![vec![0.0; 2]; n_nodes];
        for i in 0..n_nodes {
            for j in 0..2 {
                for k in 0..2 {
                    dn_dx[i][j] += dn_dxi[i][k] * jac_inv[k][j];
                }
            }
        }

        // Build B matrix (strain-displacement) and accumulate K_e = B^T * D * B * |J| * t * w
        let det_j_abs = det_j.abs();

        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let bi = [
                    [dn_dx[i][0], 0.0],
                    [0.0, dn_dx[i][1]],
                    [dn_dx[i][1], dn_dx[i][0]],
                ];
                let bj = [
                    [dn_dx[j][0], 0.0],
                    [0.0, dn_dx[j][1]],
                    [dn_dx[j][1], dn_dx[j][0]],
                ];

                // Compute Bi^T * D * Bj (2x2 result)
                let mut btdb = [[0.0; 2]; 2];
                for p in 0..2 {
                    for qq in 0..2 {
                        for r in 0..3 {
                            for s in 0..3 {
                                btdb[p][qq] += bi[r][p] * d_matrix[r][s] * bj[s][qq];
                            }
                        }
                    }
                }

                // Place into element stiffness matrix
                let factor = det_j_abs * thickness * w;
                ke[2 * i][2 * j] += btdb[0][0] * factor;
                ke[2 * i][2 * j + 1] += btdb[0][1] * factor;
                ke[2 * i + 1][2 * j] += btdb[1][0] * factor;
                ke[2 * i + 1][2 * j + 1] += btdb[1][1] * factor;
            }
        }
    }

    Ok(ke)
}

/// Computes the gradient (derivative) of the solution field at a point in an element
///
/// Given nodal solution values, computes du/dx at a specific point using
/// shape function derivatives.
///
/// # Arguments
/// * `mesh` - The mesh
/// * `element_id` - Element index
/// * `nodal_values` - Solution values at the element nodes
/// * `xi` - Evaluation point in natural coordinates
///
/// # Returns
/// Gradient vector [du/dx, du/dy, ...]
pub fn compute_gradient_at_point(
    mesh: &Mesh,
    element_id: usize,
    nodal_values: &[f64],
    xi: &[f64],
) -> FemResult<Vec<f64>> {
    if element_id >= mesh.elements.len() {
        return Err(FemError::ElementError(format!(
            "Element index {} out of range",
            element_id
        )));
    }

    let elem = &mesh.elements[element_id];
    let elem_type = ElementType::from_kind(&elem.kind);
    let n_nodes = elem.num_nodes();
    let coords = mesh.element_coords(element_id)?;

    if nodal_values.len() < n_nodes {
        return Err(FemError::ElementError(format!(
            "Need {} nodal values, got {}",
            n_nodes,
            nodal_values.len()
        )));
    }

    let dn_dxi = elem_type.derivatives(xi);
    let jac = compute_jacobian(&dn_dxi, &coords);
    let jac_inv = matrix_inverse(&jac)?;
    let dim = jac.len();

    // Transform derivatives to physical coordinates
    let mut dn_dx = vec![vec![0.0; dim]; n_nodes];
    for i in 0..n_nodes {
        for j in 0..dim {
            for k in 0..dim {
                dn_dx[i][j] += dn_dxi[i][k] * jac_inv[k][j];
            }
        }
    }

    // Compute gradient: du/dx_j = sum_i u_i * dN_i/dx_j
    let mut gradient = vec![0.0; dim];
    for j in 0..dim {
        for i in 0..n_nodes {
            gradient[j] += nodal_values[i] * dn_dx[i][j];
        }
    }

    Ok(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_shape_functions() {
        let elem = LineElement;

        // At left node (xi = -1): N1 = 1, N2 = 0
        let n = elem.evaluate(&[-1.0]);
        assert!((n[0] - 1.0).abs() < 1e-12);
        assert!((n[1] - 0.0).abs() < 1e-12);

        // At right node (xi = 1): N1 = 0, N2 = 1
        let n = elem.evaluate(&[1.0]);
        assert!((n[0] - 0.0).abs() < 1e-12);
        assert!((n[1] - 1.0).abs() < 1e-12);

        // At center (xi = 0): N1 = N2 = 0.5
        let n = elem.evaluate(&[0.0]);
        assert!((n[0] - 0.5).abs() < 1e-12);
        assert!((n[1] - 0.5).abs() < 1e-12);

        // Partition of unity
        let n = elem.evaluate(&[0.3]);
        assert!((n[0] + n[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_line_derivatives() {
        let elem = LineElement;
        let dn = elem.derivatives(&[0.0]);
        assert!((dn[0][0] - (-0.5)).abs() < 1e-12);
        assert!((dn[1][0] - 0.5).abs() < 1e-12);

        // Derivatives should be constant for linear element
        let dn2 = elem.derivatives(&[0.7]);
        assert!((dn2[0][0] - dn[0][0]).abs() < 1e-12);
    }

    #[test]
    fn test_tri_shape_functions() {
        let elem = TriElement;

        // At node 1 (0, 0): N1 = 1
        let n = elem.evaluate(&[0.0, 0.0]);
        assert!((n[0] - 1.0).abs() < 1e-12);
        assert!((n[1] - 0.0).abs() < 1e-12);
        assert!((n[2] - 0.0).abs() < 1e-12);

        // At node 2 (1, 0): N2 = 1
        let n = elem.evaluate(&[1.0, 0.0]);
        assert!((n[1] - 1.0).abs() < 1e-12);

        // At node 3 (0, 1): N3 = 1
        let n = elem.evaluate(&[0.0, 1.0]);
        assert!((n[2] - 1.0).abs() < 1e-12);

        // Partition of unity at centroid
        let n = elem.evaluate(&[1.0 / 3.0, 1.0 / 3.0]);
        assert!((n[0] + n[1] + n[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quad_shape_functions() {
        let elem = QuadElement;

        // At node 1 (-1, -1): N1 = 1
        let n = elem.evaluate(&[-1.0, -1.0]);
        assert!((n[0] - 1.0).abs() < 1e-12);
        assert!((n[1] - 0.0).abs() < 1e-12);
        assert!((n[2] - 0.0).abs() < 1e-12);
        assert!((n[3] - 0.0).abs() < 1e-12);

        // Partition of unity at an arbitrary point
        let n = elem.evaluate(&[0.3, -0.2]);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);

        // All 0.25 at center (0, 0)
        let n = elem.evaluate(&[0.0, 0.0]);
        for val in &n {
            assert!((val - 0.25).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gauss_quadrature_line() {
        // 2-point Gauss should integrate degree-3 polynomials exactly
        let gq = GaussQuadrature::line(2).expect("should create 2-point rule");
        assert_eq!(gq.num_points(), 2);

        // Integrate f(x) = x^3 on [-1, 1] (should be 0)
        let integral: f64 = gq
            .points
            .iter()
            .zip(gq.weights.iter())
            .map(|(p, w)| p[0].powi(3) * w)
            .sum();
        assert!(integral.abs() < 1e-12);

        // Integrate f(x) = x^2 on [-1, 1] (should be 2/3)
        let integral: f64 = gq
            .points
            .iter()
            .zip(gq.weights.iter())
            .map(|(p, w)| p[0].powi(2) * w)
            .sum();
        assert!((integral - 2.0 / 3.0).abs() < 1e-12);

        // Weights sum to 2 (length of interval [-1, 1])
        let wsum: f64 = gq.weights.iter().sum();
        assert!((wsum - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_gauss_quadrature_triangle() {
        // 3-point rule should integrate linears exactly
        let gq = GaussQuadrature::triangle(3).expect("should create 3-point rule");
        assert_eq!(gq.num_points(), 3);

        // Integrate f(x,y) = 1 over reference triangle (area = 0.5)
        let integral: f64 = gq.weights.iter().sum();
        assert!((integral - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_gauss_quadrature_quad() {
        // 2x2 rule on [-1,1]^2
        let gq = GaussQuadrature::quad(2).expect("should create 2x2 rule");
        assert_eq!(gq.num_points(), 4);

        // Integrate f(x,y) = 1 over [-1,1]^2 (area = 4)
        let integral: f64 = gq.weights.iter().sum();
        assert!((integral - 4.0).abs() < 1e-12);

        // 3x3 rule
        let gq3 = GaussQuadrature::quad(3).expect("should create 3x3 rule");
        assert_eq!(gq3.num_points(), 9);
        let integral3: f64 = gq3.weights.iter().sum();
        assert!((integral3 - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_gauss_5point_1d() {
        // 5-point rule should integrate degree-9 polynomials exactly
        let gq = GaussQuadrature::line(5).expect("should create 5-point rule");
        assert_eq!(gq.num_points(), 5);

        // Weights sum to 2
        let wsum: f64 = gq.weights.iter().sum();
        assert!((wsum - 2.0).abs() < 1e-12);

        // Integrate x^8 on [-1, 1] exactly: integral = 2/9
        let integral: f64 = gq
            .points
            .iter()
            .zip(gq.weights.iter())
            .map(|(p, w)| p[0].powi(8) * w)
            .sum();
        assert!((integral - 2.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_quadrature_order() {
        assert!(GaussQuadrature::line(0).is_err());
        assert!(GaussQuadrature::line(6).is_err());
        assert!(GaussQuadrature::triangle(2).is_err());
    }

    #[test]
    fn test_jacobian_line() {
        // Line element from x=1 to x=3: length = 2, J = dx/dxi = 1
        let dshape = vec![vec![-0.5], vec![0.5]];
        let coords = vec![vec![1.0], vec![3.0]];
        let jac = compute_jacobian(&dshape, &coords);
        assert!((jac[0][0] - 1.0).abs() < 1e-12); // (3-1)/2 = 1
    }

    #[test]
    fn test_jacobian_quad() {
        // Unit square quad: (0,0), (1,0), (1,1), (0,1) mapped from [-1,1]^2
        let elem = QuadElement;
        let xi = vec![0.0, 0.0];
        let dn = elem.derivatives(&xi);
        let coords = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 1.0],
        ];
        let jac = compute_jacobian(&dn, &coords);
        // For unit square, J = [[0.5, 0], [0, 0.5]]
        assert!((jac[0][0] - 0.5).abs() < 1e-12);
        assert!(jac[0][1].abs() < 1e-12);
        assert!(jac[1][0].abs() < 1e-12);
        assert!((jac[1][1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_determinant() {
        let m2 = vec![vec![2.0, 3.0], vec![1.0, 4.0]];
        let det = matrix_determinant(&m2).expect("should compute det");
        assert!((det - 5.0).abs() < 1e-12);

        let m3 = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0, 4.0],
            vec![5.0, 6.0, 0.0],
        ];
        let det = matrix_determinant(&m3).expect("should compute det");
        assert!((det - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_matrix_inverse() {
        let m = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let inv = matrix_inverse(&m).expect("should compute inverse");
        // Check M * M^-1 = I
        let id00 = m[0][0] * inv[0][0] + m[0][1] * inv[1][0];
        let id01 = m[0][0] * inv[0][1] + m[0][1] * inv[1][1];
        let id10 = m[1][0] * inv[0][0] + m[1][1] * inv[1][0];
        let id11 = m[1][0] * inv[0][1] + m[1][1] * inv[1][1];
        assert!((id00 - 1.0).abs() < 1e-12);
        assert!(id01.abs() < 1e-12);
        assert!(id10.abs() < 1e-12);
        assert!((id11 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_element_stiffness_1d() {
        // Single element [0, 1]: K should be [[k/L, -k/L], [-k/L, k/L]]
        let mesh = Mesh::generate_1d(0.0, 1.0, 1).expect("mesh gen should succeed");
        let k = 2.0;
        let (ke, fe) =
            compute_element_stiffness(&mesh, 0, k, &|_| 0.0, 2).expect("stiffness should succeed");

        // K = k/L * [[1, -1], [-1, 1]]  where L = 1
        assert!((ke[0][0] - k).abs() < 1e-10);
        assert!((ke[0][1] + k).abs() < 1e-10);
        assert!((ke[1][0] + k).abs() < 1e-10);
        assert!((ke[1][1] - k).abs() < 1e-10);

        // With zero source, load vector should be zero
        assert!(fe[0].abs() < 1e-12);
        assert!(fe[1].abs() < 1e-12);
    }

    #[test]
    fn test_element_stiffness_1d_with_source() {
        // Single element [0, 1] with constant source f=1
        let mesh = Mesh::generate_1d(0.0, 1.0, 1).expect("mesh gen should succeed");
        let (_, fe) = compute_element_stiffness(&mesh, 0, 1.0, &|_| 1.0, 2)
            .expect("stiffness should succeed");

        // For uniform source f=1 on [0, 1]:
        // f_1 = integral(1 * N_1, 0, 1) = 0.5
        // f_2 = integral(1 * N_2, 0, 1) = 0.5
        assert!((fe[0] - 0.5).abs() < 1e-10);
        assert!((fe[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_element_stiffness_symmetry() {
        // Element stiffness matrix should be symmetric
        let mesh = Mesh::generate_2d_triangular(0.0, 1.0, 0.0, 1.0, 2, 2)
            .expect("mesh gen should succeed");
        let (ke, _) = compute_element_stiffness(&mesh, 0, 1.0, &|_| 0.0, 3)
            .expect("stiffness should succeed");

        for i in 0..ke.len() {
            for j in 0..ke[i].len() {
                assert!(
                    (ke[i][j] - ke[j][i]).abs() < 1e-12,
                    "K[{}][{}] = {} != K[{}][{}] = {}",
                    i,
                    j,
                    ke[i][j],
                    j,
                    i,
                    ke[j][i]
                );
            }
        }
    }

    #[test]
    fn test_element_stiffness_row_sum_zero() {
        // Row sums of stiffness matrix should be zero (rigid body mode)
        let mesh = Mesh::generate_1d(0.0, 2.0, 1).expect("mesh gen should succeed");
        let (ke, _) = compute_element_stiffness(&mesh, 0, 3.5, &|_| 0.0, 2)
            .expect("stiffness should succeed");

        for i in 0..ke.len() {
            let row_sum: f64 = ke[i].iter().sum();
            assert!(
                row_sum.abs() < 1e-10,
                "Row {} sum is {} (should be 0)",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_element_type_from_kind() {
        let line = ElementType::from_kind(&ElementKind::Line2);
        assert_eq!(line.node_count(), 2);
        assert_eq!(line.spatial_dimension(), 1);

        let tri = ElementType::from_kind(&ElementKind::Triangle3);
        assert_eq!(tri.node_count(), 3);
        assert_eq!(tri.spatial_dimension(), 2);

        let quad = ElementType::from_kind(&ElementKind::Quad4);
        assert_eq!(quad.node_count(), 4);
        assert_eq!(quad.spatial_dimension(), 2);
    }
}
