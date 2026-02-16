//! Computational Geometry module for NumRS2
//!
//! This module provides comprehensive computational geometry algorithms and data structures
//! including convex hulls, Delaunay triangulation, Voronoi diagrams, and polygon operations.
//!
//! # Module Organization
//!
//! - [`convex_hull`]: Convex hull algorithms (Graham scan, gift wrapping)
//! - [`triangulation`]: Delaunay triangulation and mesh generation
//! - [`voronoi`]: Voronoi diagram computation and Lloyd's relaxation
//! - [`polygon`]: Polygon operations (area, centroid, containment, clipping)
//!
//! # Core Types
//!
//! - [`Point2D`]: 2D point with x, y coordinates
//! - [`Point3D`]: 3D point with x, y, z coordinates
//! - [`Polygon`]: A polygon defined by an ordered sequence of vertices
//! - [`Triangle`]: A triangle defined by three vertex indices
//! - [`Edge`]: An edge defined by two vertex indices
//!
//! # Examples
//!
//! ```rust
//! use numrs2::new_modules::geometry::{Point2D, Polygon};
//!
//! let vertices = vec![
//!     Point2D::new(0.0, 0.0),
//!     Point2D::new(4.0, 0.0),
//!     Point2D::new(4.0, 3.0),
//!     Point2D::new(0.0, 3.0),
//! ];
//! let poly = Polygon::new(vertices);
//! let area = poly.area();
//! assert!((area - 12.0).abs() < 1e-10);
//! ```
//!
//! # SCIRS2 Integration Policy
//!
//! This module follows strict SCIRS2 ecosystem policies:
//! - Array operations via `scirs2_core::ndarray`
//! - No direct external dependencies
//! - Pure Rust implementation

pub mod convex_hull;
pub mod polygon;
pub mod triangulation;
pub mod voronoi;

// Re-export core types
pub use convex_hull::{
    convex_hull_area, convex_hull_merge, convex_hull_perimeter, gift_wrapping, graham_scan,
    monotone_chain, point_in_convex_hull,
};
pub use polygon::{
    convex_polygon_union, convex_polygons_intersect, line_segment_intersection,
    minimum_bounding_rectangle, point_in_polygon, polygon_area, polygon_centroid,
    polygon_clipping_sutherland_hodgman, polygon_is_convex, polygon_is_simple, polygon_perimeter,
    polygon_signed_area, polygon_simplify_rdp, IntersectionResult,
};
pub use triangulation::{
    constrained_delaunay, delaunay_triangulation, mesh_refine, point_location,
    triangle_aspect_ratio, triangle_minimum_angle, DelaunayTriangulation,
};
pub use voronoi::{
    lloyd_relaxation, nearest_neighbor_voronoi, voronoi_from_delaunay, VoronoiDiagram, VoronoiEdge,
};

use std::fmt;
use std::hash::{Hash, Hasher};
use thiserror::Error;

/// Tolerance for floating-point comparisons in geometry operations
pub const GEOMETRY_EPSILON: f64 = 1e-10;

/// Error type for geometry operations
#[derive(Error, Debug, Clone)]
pub enum GeometryError {
    /// Insufficient number of points for the requested operation
    #[error("Insufficient points: need at least {needed}, got {got}")]
    InsufficientPoints {
        /// Minimum number of points required
        needed: usize,
        /// Actual number of points provided
        got: usize,
    },

    /// Degenerate input (e.g., collinear points for triangulation)
    #[error("Degenerate geometry: {0}")]
    DegenerateGeometry(String),

    /// Invalid polygon (e.g., self-intersecting)
    #[error("Invalid polygon: {0}")]
    InvalidPolygon(String),

    /// Point not found in the given structure
    #[error("Point not found: {0}")]
    PointNotFound(String),

    /// Numerical precision issue
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Index out of bounds
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),

    /// General computation error
    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Result type alias for geometry operations
pub type GeometryResult<T> = std::result::Result<T, GeometryError>;

/// A 2D point with x, y coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

impl Point2D {
    /// Creates a new 2D point
    ///
    /// # Arguments
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Returns the origin point (0, 0)
    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Computes the Euclidean distance to another point
    ///
    /// # Arguments
    /// * `other` - The other point
    ///
    /// # Returns
    /// The Euclidean distance between the two points
    pub fn distance(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Computes the squared Euclidean distance to another point
    ///
    /// This is more efficient than `distance` when only relative distances are needed.
    ///
    /// # Arguments
    /// * `other` - The other point
    ///
    /// # Returns
    /// The squared Euclidean distance
    pub fn distance_squared(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Computes the dot product with another point (treated as a vector from origin)
    ///
    /// # Arguments
    /// * `other` - The other point/vector
    ///
    /// # Returns
    /// The dot product
    pub fn dot(&self, other: &Point2D) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Computes the cross product (z-component) with another point (treated as vectors from origin)
    ///
    /// The cross product of 2D vectors (a, b) and (c, d) is ad - bc.
    /// This is equivalent to the signed area of the parallelogram formed by the two vectors.
    ///
    /// # Arguments
    /// * `other` - The other point/vector
    ///
    /// # Returns
    /// The z-component of the cross product
    pub fn cross(&self, other: &Point2D) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Subtracts another point from this one (vector subtraction)
    ///
    /// # Arguments
    /// * `other` - The point to subtract
    ///
    /// # Returns
    /// A new point representing the difference vector
    pub fn sub(&self, other: &Point2D) -> Point2D {
        Point2D::new(self.x - other.x, self.y - other.y)
    }

    /// Adds another point to this one (vector addition)
    ///
    /// # Arguments
    /// * `other` - The point to add
    ///
    /// # Returns
    /// A new point representing the sum vector
    pub fn add(&self, other: &Point2D) -> Point2D {
        Point2D::new(self.x + other.x, self.y + other.y)
    }

    /// Scales the point by a scalar factor
    ///
    /// # Arguments
    /// * `factor` - The scaling factor
    ///
    /// # Returns
    /// A new scaled point
    pub fn scale(&self, factor: f64) -> Point2D {
        Point2D::new(self.x * factor, self.y * factor)
    }

    /// Computes the magnitude (length) of the point vector from origin
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Returns the angle of the vector from origin in radians
    pub fn angle(&self) -> f64 {
        self.y.atan2(self.x)
    }
}

impl fmt::Display for Point2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// A 3D point with x, y, z coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Z coordinate
    pub z: f64,
}

impl Point3D {
    /// Creates a new 3D point
    ///
    /// # Arguments
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    /// * `z` - Z coordinate
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Returns the origin point (0, 0, 0)
    pub fn origin() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Computes the Euclidean distance to another point
    pub fn distance(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Computes the squared Euclidean distance to another point
    pub fn distance_squared(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Computes the dot product with another 3D point
    pub fn dot(&self, other: &Point3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Computes the cross product with another 3D point
    pub fn cross(&self, other: &Point3D) -> Point3D {
        Point3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Subtracts another point from this one
    pub fn sub(&self, other: &Point3D) -> Point3D {
        Point3D::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Adds another point to this one
    pub fn add(&self, other: &Point3D) -> Point3D {
        Point3D::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Scales the point by a scalar factor
    pub fn scale(&self, factor: f64) -> Point3D {
        Point3D::new(self.x * factor, self.y * factor, self.z * factor)
    }

    /// Computes the magnitude (length) of the point vector from origin
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Projects this 3D point onto the XY plane, returning a 2D point
    pub fn to_2d(&self) -> Point2D {
        Point2D::new(self.x, self.y)
    }
}

impl fmt::Display for Point3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

/// A triangle defined by three vertex indices into a point array
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Triangle {
    /// Index of the first vertex
    pub a: usize,
    /// Index of the second vertex
    pub b: usize,
    /// Index of the third vertex
    pub c: usize,
}

impl Triangle {
    /// Creates a new triangle from three vertex indices
    pub fn new(a: usize, b: usize, c: usize) -> Self {
        Self { a, b, c }
    }

    /// Returns the three vertex indices as a tuple
    pub fn vertices(&self) -> (usize, usize, usize) {
        (self.a, self.b, self.c)
    }

    /// Returns the three edges of the triangle
    pub fn edges(&self) -> [Edge; 3] {
        [
            Edge::new(self.a, self.b),
            Edge::new(self.b, self.c),
            Edge::new(self.c, self.a),
        ]
    }

    /// Checks if this triangle contains a given vertex index
    pub fn contains_vertex(&self, idx: usize) -> bool {
        self.a == idx || self.b == idx || self.c == idx
    }

    /// Checks if this triangle shares an edge with another triangle
    pub fn shares_edge_with(&self, other: &Triangle) -> bool {
        let my_edges = self.edges();
        let other_edges = other.edges();
        for me in &my_edges {
            for oe in &other_edges {
                if me == oe {
                    return true;
                }
            }
        }
        false
    }
}

impl fmt::Display for Triangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Triangle({}, {}, {})", self.a, self.b, self.c)
    }
}

/// An edge defined by two vertex indices
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    /// Index of the first vertex
    pub start: usize,
    /// Index of the second vertex
    pub end: usize,
}

impl Edge {
    /// Creates a new edge from two vertex indices
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Returns the edge as a canonically ordered pair (smaller index first)
    pub fn canonical(&self) -> (usize, usize) {
        if self.start <= self.end {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        }
    }

    /// Returns the other endpoint given one endpoint
    pub fn other(&self, vertex: usize) -> Option<usize> {
        if vertex == self.start {
            Some(self.end)
        } else if vertex == self.end {
            Some(self.start)
        } else {
            None
        }
    }
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.canonical() == other.canonical()
    }
}

impl Eq for Edge {}

impl Hash for Edge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let (a, b) = self.canonical();
        a.hash(state);
        b.hash(state);
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Edge({}, {})", self.start, self.end)
    }
}

/// A polygon defined by an ordered sequence of vertices
#[derive(Debug, Clone)]
pub struct Polygon {
    /// The vertices of the polygon in order (counter-clockwise for positive area)
    pub vertices: Vec<Point2D>,
}

impl Polygon {
    /// Creates a new polygon from a list of vertices
    ///
    /// # Arguments
    /// * `vertices` - Ordered list of vertices
    pub fn new(vertices: Vec<Point2D>) -> Self {
        Self { vertices }
    }

    /// Returns the number of vertices in the polygon
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the number of edges in the polygon
    pub fn num_edges(&self) -> usize {
        self.vertices.len()
    }

    /// Checks if the polygon has at least 3 vertices
    pub fn is_valid(&self) -> bool {
        self.vertices.len() >= 3
    }

    /// Computes the signed area of the polygon using the shoelace formula
    ///
    /// Positive area means counter-clockwise vertex ordering,
    /// negative area means clockwise vertex ordering.
    ///
    /// # Returns
    /// The signed area of the polygon
    pub fn signed_area(&self) -> f64 {
        polygon::polygon_signed_area(&self.vertices)
    }

    /// Computes the (unsigned) area of the polygon using the shoelace formula
    ///
    /// # Returns
    /// The area of the polygon
    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    /// Computes the centroid of the polygon
    ///
    /// # Returns
    /// The centroid point, or an error if the polygon is degenerate
    pub fn centroid(&self) -> GeometryResult<Point2D> {
        polygon::polygon_centroid(&self.vertices)
    }

    /// Computes the perimeter of the polygon
    ///
    /// # Returns
    /// The perimeter length
    pub fn perimeter(&self) -> f64 {
        polygon::polygon_perimeter(&self.vertices)
    }

    /// Checks if a point is inside the polygon using the ray casting algorithm
    ///
    /// # Arguments
    /// * `point` - The point to test
    ///
    /// # Returns
    /// `true` if the point is inside the polygon
    pub fn contains(&self, point: &Point2D) -> bool {
        polygon::point_in_polygon(point, &self.vertices)
    }

    /// Checks if the polygon is convex
    ///
    /// # Returns
    /// `true` if the polygon is convex
    pub fn is_convex(&self) -> bool {
        polygon::polygon_is_convex(&self.vertices)
    }
}

impl fmt::Display for Polygon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Polygon({} vertices)", self.vertices.len())
    }
}

/// Computes the orientation of three points
///
/// Returns:
/// - Positive value: counter-clockwise (left turn)
/// - Negative value: clockwise (right turn)
/// - Zero: collinear
///
/// The magnitude is twice the signed area of the triangle formed by the three points.
///
/// # Arguments
/// * `p` - First point
/// * `q` - Second point
/// * `r` - Third point
///
/// # Returns
/// The signed area * 2 of the triangle (p, q, r)
pub fn orientation(p: &Point2D, q: &Point2D, r: &Point2D) -> f64 {
    (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)
}

/// Determines if three points are in counter-clockwise order
///
/// # Arguments
/// * `p` - First point
/// * `q` - Second point
/// * `r` - Third point
pub fn is_ccw(p: &Point2D, q: &Point2D, r: &Point2D) -> bool {
    orientation(p, q, r) > GEOMETRY_EPSILON
}

/// Determines if three points are collinear (within tolerance)
///
/// # Arguments
/// * `p` - First point
/// * `q` - Second point
/// * `r` - Third point
pub fn is_collinear(p: &Point2D, q: &Point2D, r: &Point2D) -> bool {
    orientation(p, q, r).abs() <= GEOMETRY_EPSILON
}

/// Computes the circumcircle of three 2D points
///
/// Returns the center and squared radius of the circumscribed circle.
///
/// # Arguments
/// * `a` - First point
/// * `b` - Second point
/// * `c` - Third point
///
/// # Returns
/// `(center, radius_squared)` or error if the points are collinear
pub fn circumcircle(a: &Point2D, b: &Point2D, c: &Point2D) -> GeometryResult<(Point2D, f64)> {
    let ax = a.x;
    let ay = a.y;
    let bx = b.x;
    let by = b.y;
    let cx = c.x;
    let cy = c.y;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

    if d.abs() < GEOMETRY_EPSILON {
        return Err(GeometryError::DegenerateGeometry(
            "Points are collinear, circumcircle is undefined".to_string(),
        ));
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;

    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;

    let center = Point2D::new(ux, uy);
    let r_sq = center.distance_squared(a);

    Ok((center, r_sq))
}

/// Checks if a point is inside the circumcircle of three other points
///
/// # Arguments
/// * `p` - The point to test
/// * `a`, `b`, `c` - The three points defining the circumcircle
///
/// # Returns
/// `true` if `p` is strictly inside the circumcircle of (a, b, c)
pub fn in_circumcircle(p: &Point2D, a: &Point2D, b: &Point2D, c: &Point2D) -> bool {
    // Use the determinant method for robustness
    // The point p is inside the circumcircle of (a, b, c) if:
    // | ax-px  ay-py  (ax-px)^2+(ay-py)^2 |
    // | bx-px  by-py  (bx-px)^2+(by-py)^2 | > 0
    // | cx-px  cy-py  (cx-px)^2+(cy-py)^2 |
    // (assuming a, b, c are in counter-clockwise order)

    let adx = a.x - p.x;
    let ady = a.y - p.y;
    let bdx = b.x - p.x;
    let bdy = b.y - p.y;
    let cdx = c.x - p.x;
    let cdy = c.y - p.y;

    let ab_lift = adx * adx + ady * ady;
    let bb_lift = bdx * bdx + bdy * bdy;
    let cb_lift = cdx * cdx + cdy * cdy;

    let det = adx * (bdy * cb_lift - cdy * bb_lift) - ady * (bdx * cb_lift - cdx * bb_lift)
        + ab_lift * (bdx * cdy - cdx * bdy);

    // If (a, b, c) are in CCW order, det > 0 means p is inside
    // If (a, b, c) are in CW order, det < 0 means p is inside
    let orient = orientation(a, b, c);
    if orient > 0.0 {
        det > GEOMETRY_EPSILON
    } else {
        det < -GEOMETRY_EPSILON
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_basic() {
        let p = Point2D::new(3.0, 4.0);
        assert!((p.magnitude() - 5.0).abs() < 1e-10);

        let q = Point2D::new(0.0, 0.0);
        assert!((p.distance(&q) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point2d_operations() {
        let a = Point2D::new(1.0, 2.0);
        let b = Point2D::new(3.0, 4.0);

        let sum = a.add(&b);
        assert!((sum.x - 4.0).abs() < 1e-10);
        assert!((sum.y - 6.0).abs() < 1e-10);

        let diff = b.sub(&a);
        assert!((diff.x - 2.0).abs() < 1e-10);
        assert!((diff.y - 2.0).abs() < 1e-10);

        let scaled = a.scale(2.0);
        assert!((scaled.x - 2.0).abs() < 1e-10);
        assert!((scaled.y - 4.0).abs() < 1e-10);

        assert!((a.dot(&b) - 11.0).abs() < 1e-10);
        assert!((a.cross(&b) - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_basic() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        let q = Point3D::origin();

        let dist = p.distance(&q);
        assert!((dist - (14.0_f64).sqrt()).abs() < 1e-10);

        let p2d = p.to_2d();
        assert!((p2d.x - 1.0).abs() < 1e-10);
        assert!((p2d.y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_cross_product() {
        let a = Point3D::new(1.0, 0.0, 0.0);
        let b = Point3D::new(0.0, 1.0, 0.0);
        let c = a.cross(&b);
        assert!((c.x - 0.0).abs() < 1e-10);
        assert!((c.y - 0.0).abs() < 1e-10);
        assert!((c.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_basic() {
        let t = Triangle::new(0, 1, 2);
        assert_eq!(t.vertices(), (0, 1, 2));
        assert!(t.contains_vertex(1));
        assert!(!t.contains_vertex(3));

        let edges = t.edges();
        assert_eq!(edges[0], Edge::new(0, 1));
        assert_eq!(edges[1], Edge::new(1, 2));
        assert_eq!(edges[2], Edge::new(2, 0));
    }

    #[test]
    fn test_edge_equality() {
        let e1 = Edge::new(0, 1);
        let e2 = Edge::new(1, 0);
        assert_eq!(e1, e2);

        let e3 = Edge::new(0, 2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_orientation() {
        let p = Point2D::new(0.0, 0.0);
        let q = Point2D::new(1.0, 0.0);
        let r = Point2D::new(0.0, 1.0);

        // CCW
        assert!(orientation(&p, &q, &r) > 0.0);
        // CW
        assert!(orientation(&p, &r, &q) < 0.0);

        // Collinear
        let s = Point2D::new(2.0, 0.0);
        assert!(is_collinear(&p, &q, &s));
    }

    #[test]
    fn test_circumcircle() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(1.0, 0.0);
        let c = Point2D::new(0.5, 0.5_f64.sqrt());

        let result = circumcircle(&a, &b, &c);
        assert!(result.is_ok());

        let (center, r_sq) = result.expect("circumcircle should succeed");
        // Equilateral triangle => circumcenter at (0.5, ~0.2887)
        assert!((center.x - 0.5).abs() < 1e-6);
        // Verify all points are equidistant from center
        let d_a = center.distance_squared(&a);
        let d_b = center.distance_squared(&b);
        let d_c = center.distance_squared(&c);
        assert!((d_a - d_b).abs() < 1e-8);
        assert!((d_b - d_c).abs() < 1e-8);
        assert!((d_a - r_sq).abs() < 1e-8);
    }

    #[test]
    fn test_in_circumcircle() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(4.0, 0.0);
        let c = Point2D::new(0.0, 4.0);

        // Point inside the circumcircle
        let inside = Point2D::new(1.0, 1.0);
        assert!(in_circumcircle(&inside, &a, &b, &c));

        // Point outside the circumcircle
        let outside = Point2D::new(10.0, 10.0);
        assert!(!in_circumcircle(&outside, &a, &b, &c));
    }

    #[test]
    fn test_polygon_basic() {
        let poly = Polygon::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 3.0),
            Point2D::new(0.0, 3.0),
        ]);
        assert!(poly.is_valid());
        assert_eq!(poly.num_vertices(), 4);
        assert_eq!(poly.num_edges(), 4);
        assert!((poly.area() - 12.0).abs() < 1e-10);
        assert!(poly.is_convex());
    }

    #[test]
    fn test_triangle_shares_edge() {
        let t1 = Triangle::new(0, 1, 2);
        let t2 = Triangle::new(1, 2, 3);
        let t3 = Triangle::new(3, 4, 5);

        assert!(t1.shares_edge_with(&t2));
        assert!(!t1.shares_edge_with(&t3));
    }

    #[test]
    fn test_display_traits() {
        let p2 = Point2D::new(1.5, 2.5);
        assert_eq!(format!("{}", p2), "(1.5, 2.5)");

        let p3 = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(format!("{}", p3), "(1, 2, 3)");

        let t = Triangle::new(0, 1, 2);
        assert_eq!(format!("{}", t), "Triangle(0, 1, 2)");

        let e = Edge::new(0, 1);
        assert_eq!(format!("{}", e), "Edge(0, 1)");
    }
}
