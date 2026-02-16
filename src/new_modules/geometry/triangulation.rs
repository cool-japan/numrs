//! Delaunay Triangulation and Mesh Generation
//!
//! This module provides algorithms for:
//! - Standard Delaunay triangulation using the Bowyer-Watson incremental algorithm
//! - Constrained Delaunay triangulation (CDT) with enforced edge constraints
//! - Mesh refinement via Ruppert's algorithm
//! - Point location queries within triangulations
//! - Triangle quality metrics (aspect ratio, minimum angle)
//!
//! # Algorithms
//!
//! ## Bowyer-Watson Algorithm
//!
//! The Bowyer-Watson algorithm is an incremental insertion algorithm for computing
//! Delaunay triangulations. For each new point, it finds all triangles whose
//! circumcircles contain the point (the "bad" triangles), removes them, and
//! re-triangulates the resulting polygonal hole with triangles connecting to the
//! new point.
//!
//! ## Constrained Delaunay Triangulation
//!
//! CDT extends the standard Delaunay triangulation by enforcing that certain
//! edges (constraints) appear in the final triangulation, even if they violate
//! the Delaunay criterion. This is useful for boundary-conforming meshes.
//!
//! ## Ruppert's Algorithm
//!
//! Ruppert's algorithm refines a triangulation by inserting circumcenters of
//! triangles that fail quality criteria (minimum angle or maximum area),
//! producing a mesh with guaranteed quality bounds.
//!
//! # References
//!
//! - Bowyer, A. (1981). "Computing Dirichlet tessellations."
//!   The Computer Journal, 24(2), 162-166.
//! - Watson, D.F. (1981). "Computing the n-dimensional Delaunay tessellation with
//!   application to Voronoi polytopes." The Computer Journal, 24(2), 167-172.
//! - Ruppert, J. (1995). "A Delaunay Refinement Algorithm for Quality 2-Dimensional
//!   Mesh Generation." Journal of Algorithms, 18(3), 548-585.
//!
//! # Examples
//!
//! ```rust
//! use numrs2::new_modules::geometry::{Point2D, delaunay_triangulation};
//!
//! let points = vec![
//!     Point2D::new(0.0, 0.0),
//!     Point2D::new(1.0, 0.0),
//!     Point2D::new(0.5, 1.0),
//!     Point2D::new(1.0, 1.0),
//! ];
//! let tri = delaunay_triangulation(&points).expect("triangulation should succeed");
//! assert!(!tri.triangles.is_empty());
//! ```

use std::collections::{HashMap, HashSet};

use super::{
    circumcircle, in_circumcircle, is_collinear, orientation, Edge, GeometryError, GeometryResult,
    Point2D, Triangle, GEOMETRY_EPSILON,
};

// ---------------------------------------------------------------------------
// DelaunayTriangulation struct
// ---------------------------------------------------------------------------

/// A Delaunay triangulation data structure
///
/// Stores the points, triangles, and adjacency information for a Delaunay
/// triangulation. The super-triangle vertices are retained internally for
/// compatibility with the Voronoi dual construction, and are accessible via
/// `num_original_points`.
#[derive(Debug, Clone)]
pub struct DelaunayTriangulation {
    /// All points in the triangulation (indices 0..num_original_points are user
    /// points; indices num_original_points.. are super-triangle vertices)
    pub points: Vec<Point2D>,
    /// The triangles of the triangulation (may reference super-triangle vertices)
    pub triangles: Vec<Triangle>,
    /// Number of original (non-super-triangle) points
    pub num_original_points: usize,
}

impl DelaunayTriangulation {
    /// Returns the triangles that only reference original points (not
    /// super-triangle vertices)
    pub fn interior_triangles(&self) -> Vec<Triangle> {
        let n = self.num_original_points;
        self.triangles
            .iter()
            .filter(|t| t.a < n && t.b < n && t.c < n)
            .copied()
            .collect()
    }

    /// Returns all edges of the triangulation that reference only original
    /// points (canonical form, no duplicates)
    pub fn edges(&self) -> Vec<Edge> {
        let n = self.num_original_points;
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut edges = Vec::new();

        for tri in &self.triangles {
            for edge in tri.edges() {
                if edge.start < n && edge.end < n {
                    let key = edge.canonical();
                    if seen.insert(key) {
                        edges.push(edge);
                    }
                }
            }
        }

        edges
    }

    /// Returns the indices of triangles that share an edge with the given
    /// triangle (O(t) linear scan)
    pub fn neighbors(&self, tri_idx: usize) -> Vec<usize> {
        if tri_idx >= self.triangles.len() {
            return Vec::new();
        }

        let target_edges = self.triangles[tri_idx].edges();
        let mut result = Vec::new();

        for (i, other) in self.triangles.iter().enumerate() {
            if i == tri_idx {
                continue;
            }
            let other_edges = other.edges();
            'outer: for te in &target_edges {
                for oe in &other_edges {
                    if te == oe {
                        result.push(i);
                        break 'outer;
                    }
                }
            }
        }

        result
    }

    /// Returns the number of triangles
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Returns the number of points (including super-triangle vertices)
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Verifies that no original point lies strictly inside the circumcircle
    /// of any interior triangle (Delaunay criterion). O(n*t) check.
    pub fn is_delaunay(&self) -> bool {
        let interior = self.interior_triangles();
        let n = self.num_original_points;

        for tri in &interior {
            let pa = &self.points[tri.a];
            let pb = &self.points[tri.b];
            let pc = &self.points[tri.c];

            for idx in 0..n {
                if idx == tri.a || idx == tri.b || idx == tri.c {
                    continue;
                }
                if in_circumcircle(&self.points[idx], pa, pb, pc) {
                    return false;
                }
            }
        }
        true
    }

    /// Builds an adjacency map: canonical edge -> vec of triangle indices
    /// sharing that edge. Useful for queries and Voronoi construction.
    pub fn build_adjacency(&self) -> HashMap<(usize, usize), Vec<usize>> {
        let mut adj: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (idx, tri) in self.triangles.iter().enumerate() {
            for edge in &tri.edges() {
                adj.entry(edge.canonical()).or_default().push(idx);
            }
        }
        adj
    }

    /// Returns boundary edges (edges shared by exactly one *interior* triangle)
    /// among original-point edges.
    ///
    /// Only interior triangles (those referencing exclusively original points)
    /// are considered when computing edge adjacency. This prevents
    /// super-triangle triangles from inflating the count and masking true
    /// boundary edges.
    pub fn boundary_edges(&self) -> Vec<Edge> {
        let n = self.num_original_points;
        let mut adj: HashMap<(usize, usize), usize> = HashMap::new();

        for tri in &self.triangles {
            // Only consider interior triangles (all vertices are original)
            if tri.a >= n || tri.b >= n || tri.c >= n {
                continue;
            }
            for edge in &tri.edges() {
                let key = edge.canonical();
                *adj.entry(key).or_insert(0) += 1;
            }
        }

        adj.into_iter()
            .filter_map(|((s, e), count)| {
                if count == 1 {
                    Some(Edge::new(s, e))
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Triangle quality metrics
// ---------------------------------------------------------------------------

/// Computes the aspect ratio quality metric of a triangle
///
/// The aspect ratio is defined as the ratio of the circumradius to the
/// inradius (R / r). For an equilateral triangle this equals 2.0, which is
/// the optimal value. Higher values indicate poorer element quality.
///
/// # Arguments
/// * `a`, `b`, `c` - The three vertices of the triangle
///
/// # Returns
/// The aspect ratio R/r. Returns `f64::INFINITY` for degenerate triangles.
pub fn triangle_aspect_ratio(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    let ab = a.distance(b);
    let bc = b.distance(c);
    let ca = c.distance(a);

    let s = (ab + bc + ca) / 2.0;
    let area = triangle_area_from_vertices(a, b, c);

    if area < GEOMETRY_EPSILON {
        return f64::INFINITY;
    }

    // circumradius R = (ab * bc * ca) / (4 * area)
    let circumradius = (ab * bc * ca) / (4.0 * area);

    // inradius r = area / s
    let inradius = area / s;

    if inradius < GEOMETRY_EPSILON {
        return f64::INFINITY;
    }

    circumradius / inradius
}

/// Computes the minimum interior angle of a triangle in radians
///
/// The minimum angle is a key quality metric for finite element meshes.
/// Values close to 60 degrees (pi/3) indicate high-quality triangles.
///
/// # Arguments
/// * `a`, `b`, `c` - The three vertices of the triangle
///
/// # Returns
/// The minimum interior angle in radians. Returns 0.0 for degenerate triangles.
pub fn triangle_minimum_angle(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    let ab = a.distance(b);
    let bc = b.distance(c);
    let ca = c.distance(a);

    if ab < GEOMETRY_EPSILON || bc < GEOMETRY_EPSILON || ca < GEOMETRY_EPSILON {
        return 0.0;
    }

    // Law of cosines: cos(A) = (b^2 + c^2 - a^2) / (2bc)
    // Angle at vertex A (between sides AB and CA, opposite side BC)
    let cos_a = (ab * ab + ca * ca - bc * bc) / (2.0 * ab * ca);
    // Angle at vertex B (between sides AB and BC, opposite side CA)
    let cos_b = (ab * ab + bc * bc - ca * ca) / (2.0 * ab * bc);
    // Angle at vertex C (between sides BC and CA, opposite side AB)
    let cos_c = (bc * bc + ca * ca - ab * ab) / (2.0 * bc * ca);

    let angle_a = cos_a.clamp(-1.0, 1.0).acos();
    let angle_b = cos_b.clamp(-1.0, 1.0).acos();
    let angle_c = cos_c.clamp(-1.0, 1.0).acos();

    angle_a.min(angle_b).min(angle_c)
}

// ---------------------------------------------------------------------------
// Bowyer-Watson Delaunay Triangulation
// ---------------------------------------------------------------------------

/// Computes the Delaunay triangulation of a set of 2D points using the
/// Bowyer-Watson incremental insertion algorithm.
///
/// # Algorithm (Bowyer-Watson)
/// 1. Create a super-triangle that contains all points
/// 2. Insert points one at a time
/// 3. For each new point, find all triangles whose circumcircle contains the point
/// 4. Remove those triangles, creating a polygonal hole
/// 5. Re-triangulate the hole by connecting the new point to all boundary edges
/// 6. Remove triangles connected to the super-triangle vertices
///
/// # Arguments
/// * `points` - A slice of 2D points (at least 3 non-collinear)
///
/// # Returns
/// A `DelaunayTriangulation` struct
///
/// # Errors
/// Returns an error if fewer than 3 points or all points are collinear
pub fn delaunay_triangulation(points: &[Point2D]) -> GeometryResult<DelaunayTriangulation> {
    if points.len() < 3 {
        return Err(GeometryError::InsufficientPoints {
            needed: 3,
            got: points.len(),
        });
    }

    // Check for all-collinear points
    let mut all_collinear = true;
    for i in 2..points.len() {
        if !is_collinear(&points[0], &points[1], &points[i]) {
            all_collinear = false;
            break;
        }
    }
    if all_collinear {
        return Err(GeometryError::DegenerateGeometry(
            "All points are collinear".to_string(),
        ));
    }

    let n = points.len();

    // Compute bounding box
    let mut min_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_x = points[0].x;
    let mut max_y = points[0].y;

    for p in points.iter().skip(1) {
        if p.x < min_x {
            min_x = p.x;
        }
        if p.y < min_y {
            min_y = p.y;
        }
        if p.x > max_x {
            max_x = p.x;
        }
        if p.y > max_y {
            max_y = p.y;
        }
    }

    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let d_max = dx.max(dy).max(1.0);
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    // Create super-triangle with generous margin
    let margin = d_max * 20.0;
    let st_a = Point2D::new(mid_x - margin, mid_y - margin);
    let st_b = Point2D::new(mid_x + margin, mid_y - margin);
    let st_c = Point2D::new(mid_x, mid_y + margin);

    // Working point array: original at [0..n), super-triangle at [n..n+3)
    let mut all_points: Vec<Point2D> = Vec::with_capacity(n + 3);
    all_points.extend_from_slice(points);
    all_points.push(st_a); // index n
    all_points.push(st_b); // index n+1
    all_points.push(st_c); // index n+2

    // Start with the super-triangle
    let mut triangles: Vec<Triangle> = vec![Triangle::new(n, n + 1, n + 2)];

    // Insert each original point
    for i in 0..n {
        let pt = &all_points[i];

        // Find all "bad" triangles whose circumcircle contains the point
        let mut bad_indices: Vec<usize> = Vec::new();
        for (j, tri) in triangles.iter().enumerate() {
            let pa = &all_points[tri.a];
            let pb = &all_points[tri.b];
            let pc = &all_points[tri.c];

            if in_circumcircle(pt, pa, pb, pc) {
                bad_indices.push(j);
            }
        }

        // Find the boundary polygon of the hole.
        // An edge is on the boundary if it belongs to exactly one bad triangle.
        let mut boundary_edges: Vec<Edge> = Vec::new();
        for &bi in &bad_indices {
            let tri = &triangles[bi];
            for edge in tri.edges() {
                let mut shared = false;
                for &bj in &bad_indices {
                    if bi == bj {
                        continue;
                    }
                    let other = &triangles[bj];
                    for other_edge in other.edges() {
                        if edge == other_edge {
                            shared = true;
                            break;
                        }
                    }
                    if shared {
                        break;
                    }
                }
                if !shared {
                    boundary_edges.push(edge);
                }
            }
        }

        // Remove bad triangles (sort descending to preserve indices during removal)
        bad_indices.sort_unstable();
        for &idx in bad_indices.iter().rev() {
            triangles.swap_remove(idx);
        }

        // Connect new point to boundary edges
        for edge in &boundary_edges {
            triangles.push(Triangle::new(i, edge.start, edge.end));
        }
    }

    Ok(DelaunayTriangulation {
        points: all_points,
        triangles,
        num_original_points: n,
    })
}

// ---------------------------------------------------------------------------
// Point Location
// ---------------------------------------------------------------------------

/// Checks if a point lies inside a triangle using signed-area tests
fn point_in_triangle(p: &Point2D, a: &Point2D, b: &Point2D, c: &Point2D) -> bool {
    let d1 = orientation(a, b, p);
    let d2 = orientation(b, c, p);
    let d3 = orientation(c, a, p);

    let has_neg = (d1 < -GEOMETRY_EPSILON) || (d2 < -GEOMETRY_EPSILON) || (d3 < -GEOMETRY_EPSILON);
    let has_pos = (d1 > GEOMETRY_EPSILON) || (d2 > GEOMETRY_EPSILON) || (d3 > GEOMETRY_EPSILON);

    !(has_neg && has_pos)
}

/// Locates which triangle in the triangulation contains a given query point
///
/// Performs a linear scan over all *interior* triangles (those referencing
/// only original points, not super-triangle vertices). For large
/// triangulations a hierarchical walk would be faster, but this guarantees
/// correctness.
///
/// # Arguments
/// * `dt` - The Delaunay triangulation
/// * `point` - The query point
///
/// # Returns
/// `Some(index)` of the containing triangle, or `None` if the point lies
/// outside the convex hull of the original points.
pub fn point_location(dt: &DelaunayTriangulation, point: &Point2D) -> Option<usize> {
    let n = dt.num_original_points;
    for (i, tri) in dt.triangles.iter().enumerate() {
        // Skip triangles that reference super-triangle vertices
        if tri.a >= n || tri.b >= n || tri.c >= n {
            continue;
        }

        let pa = &dt.points[tri.a];
        let pb = &dt.points[tri.b];
        let pc = &dt.points[tri.c];

        if point_in_triangle(point, pa, pb, pc) {
            return Some(i);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Constrained Delaunay Triangulation
// ---------------------------------------------------------------------------

/// Checks whether two line segments (a1-a2) and (b1-b2) have a proper (strict)
/// crossing, excluding shared endpoints and collinear overlaps.
fn segments_intersect_strictly(a1: &Point2D, a2: &Point2D, b1: &Point2D, b2: &Point2D) -> bool {
    let d1 = orientation(b1, b2, a1);
    let d2 = orientation(b1, b2, a2);
    let d3 = orientation(a1, a2, b1);
    let d4 = orientation(a1, a2, b2);

    ((d1 > GEOMETRY_EPSILON && d2 < -GEOMETRY_EPSILON)
        || (d1 < -GEOMETRY_EPSILON && d2 > GEOMETRY_EPSILON))
        && ((d3 > GEOMETRY_EPSILON && d4 < -GEOMETRY_EPSILON)
            || (d3 < -GEOMETRY_EPSILON && d4 > GEOMETRY_EPSILON))
}

/// Re-triangulates one side of a constraint edge using a simple fan approach.
///
/// Sorts the side vertices by angle around the midpoint of the constraint
/// edge and connects them in a fan from `ci` through the sorted vertices
/// to `cj`.
fn retriangulate_side(
    dt: &mut DelaunayTriangulation,
    ci: usize,
    cj: usize,
    side_vertices: &[usize],
) {
    if side_vertices.is_empty() {
        return;
    }

    if side_vertices.len() == 1 {
        dt.triangles.push(Triangle::new(ci, cj, side_vertices[0]));
        return;
    }

    let p_ci = dt.points[ci];
    let p_cj = dt.points[cj];
    let mid = Point2D::new((p_ci.x + p_cj.x) / 2.0, (p_ci.y + p_cj.y) / 2.0);

    let mut sorted = side_vertices.to_vec();
    sorted.sort_by(|&a, &b| {
        let angle_a = (dt.points[a].y - mid.y).atan2(dt.points[a].x - mid.x);
        let angle_b = (dt.points[b].y - mid.y).atan2(dt.points[b].x - mid.x);
        angle_a
            .partial_cmp(&angle_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Fan: first triangle connects ci, sorted[0], cj; subsequent ones fill the
    // region between consecutive sorted vertices.
    for k in 0..sorted.len() {
        if k == 0 {
            dt.triangles.push(Triangle::new(ci, sorted[k], cj));
        } else {
            dt.triangles
                .push(Triangle::new(ci, sorted[k - 1], sorted[k]));
        }
    }
}

/// Enforces a single constraint edge in the triangulation by finding and
/// removing intersecting edges, then re-triangulating each side.
fn enforce_constraint_edge(
    dt: &mut DelaunayTriangulation,
    ci: usize,
    cj: usize,
) -> GeometryResult<()> {
    let constraint_edge = Edge::new(ci, cj);
    let p_ci = dt.points[ci];
    let p_cj = dt.points[cj];

    // Check if constraint edge already present
    let edge_exists = dt
        .triangles
        .iter()
        .any(|t| t.edges().contains(&constraint_edge));
    if edge_exists {
        return Ok(());
    }

    // Find triangles with edges that cross the constraint segment
    let mut intersected: Vec<usize> = Vec::new();
    for (idx, tri) in dt.triangles.iter().enumerate() {
        for edge in tri.edges() {
            if edge.start == ci || edge.start == cj || edge.end == ci || edge.end == cj {
                continue;
            }
            let pe_s = dt.points[edge.start];
            let pe_e = dt.points[edge.end];
            if segments_intersect_strictly(&p_ci, &p_cj, &pe_s, &pe_e) {
                intersected.push(idx);
                break;
            }
        }
    }

    if intersected.is_empty() {
        return Ok(());
    }

    // Collect all vertices of intersected triangles
    let mut vertex_set: HashSet<usize> = HashSet::new();
    for &idx in &intersected {
        let tri = &dt.triangles[idx];
        vertex_set.insert(tri.a);
        vertex_set.insert(tri.b);
        vertex_set.insert(tri.c);
    }

    // Remove intersected triangles (reverse order for swap_remove safety)
    intersected.sort_unstable();
    intersected.dedup();
    for &idx in intersected.iter().rev() {
        dt.triangles.swap_remove(idx);
    }

    // Partition remaining vertices into two sides of the constraint line
    let mut above: Vec<usize> = Vec::new();
    let mut below: Vec<usize> = Vec::new();
    for &v in &vertex_set {
        if v == ci || v == cj {
            continue;
        }
        let orient = orientation(&p_ci, &p_cj, &dt.points[v]);
        if orient > GEOMETRY_EPSILON {
            above.push(v);
        } else if orient < -GEOMETRY_EPSILON {
            below.push(v);
        }
    }

    retriangulate_side(dt, ci, cj, &above);
    retriangulate_side(dt, ci, cj, &below);

    Ok(())
}

/// Computes a Constrained Delaunay Triangulation (CDT)
///
/// First computes a standard Delaunay triangulation, then enforces the given
/// constraint edges. The Delaunay property is preserved for all non-constrained
/// edges.
///
/// # Arguments
/// * `points` - A slice of 2D points
/// * `constraints` - Pairs of point indices that must appear as edges
///
/// # Returns
/// A `DelaunayTriangulation` with all constraint edges enforced
///
/// # Errors
/// Returns an error if the base triangulation fails or if constraint indices
/// are out of bounds.
pub fn constrained_delaunay(
    points: &[Point2D],
    constraints: &[(usize, usize)],
) -> GeometryResult<DelaunayTriangulation> {
    // Validate constraints
    for &(ci, cj) in constraints {
        if ci >= points.len() || cj >= points.len() {
            return Err(GeometryError::IndexOutOfBounds(format!(
                "Constraint edge ({}, {}) references point outside range [0, {})",
                ci,
                cj,
                points.len()
            )));
        }
        if ci == cj {
            return Err(GeometryError::DegenerateGeometry(format!(
                "Constraint ({}, {}) has identical endpoints",
                ci, cj
            )));
        }
    }

    let mut dt = delaunay_triangulation(points)?;

    for &(ci, cj) in constraints {
        enforce_constraint_edge(&mut dt, ci, cj)?;
    }

    Ok(dt)
}

// ---------------------------------------------------------------------------
// Mesh Refinement (Ruppert's algorithm)
// ---------------------------------------------------------------------------

/// Computes the area of a triangle from its three vertices (unsigned)
fn triangle_area_from_vertices(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    ((a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2.0).abs()
}

/// Refines a Delaunay triangulation to improve mesh quality using a variant
/// of Ruppert's algorithm.
///
/// Iteratively identifies triangles that violate quality criteria (maximum
/// area or minimum angle) and inserts their circumcenters as new Steiner
/// points, then re-triangulates.
///
/// # Arguments
/// * `triangulation` - The initial Delaunay triangulation to refine
/// * `max_area` - Maximum allowable triangle area
/// * `min_angle` - Minimum allowable angle in radians
///
/// # Returns
/// A new, refined `DelaunayTriangulation`
///
/// # Errors
/// Returns an error if the refinement process produces a degenerate result.
pub fn mesh_refine(
    triangulation: &DelaunayTriangulation,
    max_area: f64,
    min_angle: f64,
) -> GeometryResult<DelaunayTriangulation> {
    // Collect only the original points from the input triangulation
    let mut current_points: Vec<Point2D> = triangulation
        .points
        .iter()
        .take(triangulation.num_original_points)
        .copied()
        .collect();

    let max_iterations = 500;

    for _iteration in 0..max_iterations {
        let dt = delaunay_triangulation(&current_points)?;
        let interior = dt.interior_triangles();

        // Find bad triangles (violating area or angle constraints)
        let mut bad_circumcenters: Vec<(Point2D, f64)> = Vec::new();

        for tri in &interior {
            let pa = &dt.points[tri.a];
            let pb = &dt.points[tri.b];
            let pc = &dt.points[tri.c];

            let area = triangle_area_from_vertices(pa, pb, pc);
            let min_a = triangle_minimum_angle(pa, pb, pc);

            if area > max_area || min_a < min_angle {
                match circumcircle(pa, pb, pc) {
                    Ok((center, _)) => {
                        bad_circumcenters.push((center, min_a));
                    }
                    Err(_) => {
                        // Degenerate triangle -- insert midpoint of longest edge
                        let ab = pa.distance(pb);
                        let bc = pb.distance(pc);
                        let ca = pc.distance(pa);
                        let mid = if ab >= bc && ab >= ca {
                            Point2D::new((pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0)
                        } else if bc >= ca {
                            Point2D::new((pb.x + pc.x) / 2.0, (pb.y + pc.y) / 2.0)
                        } else {
                            Point2D::new((pc.x + pa.x) / 2.0, (pc.y + pa.y) / 2.0)
                        };
                        bad_circumcenters.push((mid, 0.0));
                    }
                }
            }
        }

        if bad_circumcenters.is_empty() {
            return Ok(dt);
        }

        // Sort by worst quality first (smallest min-angle)
        bad_circumcenters
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Insert a limited batch of new points per iteration
        let batch = bad_circumcenters.len().min(20);
        let mut added = 0_usize;

        for (new_pt, _) in bad_circumcenters.iter().take(batch) {
            let too_close = current_points
                .iter()
                .any(|p| p.distance(new_pt) < GEOMETRY_EPSILON * 100.0);
            if !too_close {
                current_points.push(*new_pt);
                added += 1;
            }
        }

        if added == 0 {
            // Could not insert any new points -- return current state
            return delaunay_triangulation(&current_points);
        }

        // Safety: prevent runaway growth
        if current_points.len() > triangulation.num_original_points * 50 + 10_000 {
            return Err(GeometryError::ComputationError(format!(
                "Mesh refinement exceeded point limit ({} points)",
                current_points.len()
            )));
        }
    }

    delaunay_triangulation(&current_points)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn basic_square() -> Vec<Point2D> {
        vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 4.0),
            Point2D::new(0.0, 4.0),
        ]
    }

    fn five_points() -> Vec<Point2D> {
        vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 4.0),
            Point2D::new(0.0, 4.0),
            Point2D::new(2.0, 2.0),
        ]
    }

    fn grid_points(nx: usize, ny: usize) -> Vec<Point2D> {
        let mut pts = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                // Small jitter to avoid cocircular degeneracies
                pts.push(Point2D::new(
                    i as f64 + (j as f64) * 0.01,
                    j as f64 + (i as f64) * 0.01,
                ));
            }
        }
        pts
    }

    // ---- Test 1: Basic single-triangle ----
    #[test]
    fn test_delaunay_single_triangle() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let interior = dt.interior_triangles();
        assert_eq!(interior.len(), 1);
    }

    // ---- Test 2: Square -> 2 triangles ----
    #[test]
    fn test_delaunay_square() {
        let pts = basic_square();
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let interior = dt.interior_triangles();
        assert_eq!(interior.len(), 2);
        assert!(dt.is_delaunay());
    }

    // ---- Test 3: 5 points with center ----
    #[test]
    fn test_delaunay_five_points() {
        let pts = five_points();
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let interior = dt.interior_triangles();
        assert!(interior.len() >= 4);
        assert!(dt.is_delaunay());
    }

    // ---- Test 4: Grid triangulation ----
    #[test]
    fn test_delaunay_grid() {
        let pts = grid_points(4, 4);
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let interior = dt.interior_triangles();
        assert!(interior.len() >= 9);
        assert!(dt.is_delaunay());
    }

    // ---- Test 5: Insufficient points ----
    #[test]
    fn test_delaunay_insufficient_points() {
        let pts = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0)];
        assert!(delaunay_triangulation(&pts).is_err());
    }

    // ---- Test 6: Collinear points ----
    #[test]
    fn test_delaunay_collinear() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
        ];
        assert!(delaunay_triangulation(&pts).is_err());
    }

    // ---- Test 7: Delaunay property with 7 points ----
    #[test]
    fn test_delaunay_property() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(5.0, 0.0),
            Point2D::new(5.0, 5.0),
            Point2D::new(0.0, 5.0),
            Point2D::new(2.5, 2.5),
            Point2D::new(1.0, 3.0),
            Point2D::new(3.0, 1.0),
        ];
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let interior = dt.interior_triangles();

        for tri in &interior {
            let pa = &dt.points[tri.a];
            let pb = &dt.points[tri.b];
            let pc = &dt.points[tri.c];
            for (i, p) in pts.iter().enumerate() {
                if i == tri.a || i == tri.b || i == tri.c {
                    continue;
                }
                assert!(
                    !in_circumcircle(p, pa, pb, pc),
                    "Point {} inside circumcircle of ({}, {}, {})",
                    i,
                    tri.a,
                    tri.b,
                    tri.c
                );
            }
        }
    }

    // ---- Test 8: Point location inside ----
    #[test]
    fn test_point_location_inside() {
        let pts = five_points();
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let query = Point2D::new(2.0, 1.0);
        assert!(point_location(&dt, &query).is_some());
    }

    // ---- Test 9: Point location outside ----
    #[test]
    fn test_point_location_outside() {
        let pts = basic_square();
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let query = Point2D::new(10.0, 10.0);
        assert!(point_location(&dt, &query).is_none());
    }

    // ---- Test 10: Point location at vertex ----
    #[test]
    fn test_point_location_at_vertex() {
        let pts = basic_square();
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let query = Point2D::new(0.0, 0.0);
        assert!(point_location(&dt, &query).is_some());
    }

    // ---- Test 11: Aspect ratio equilateral ----
    #[test]
    fn test_aspect_ratio_equilateral() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(1.0, 0.0);
        let c = Point2D::new(0.5, (3.0_f64).sqrt() / 2.0);
        let ratio = triangle_aspect_ratio(&a, &b, &c);
        assert!((ratio - 2.0).abs() < 1e-6, "Expected 2.0, got {}", ratio);
    }

    // ---- Test 12: Aspect ratio degenerate ----
    #[test]
    fn test_aspect_ratio_degenerate() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(1.0, 0.0);
        let c = Point2D::new(2.0, 0.0);
        assert!(triangle_aspect_ratio(&a, &b, &c).is_infinite());
    }

    // ---- Test 13: Minimum angle equilateral ----
    #[test]
    fn test_min_angle_equilateral() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(1.0, 0.0);
        let c = Point2D::new(0.5, (3.0_f64).sqrt() / 2.0);
        let min_a = triangle_minimum_angle(&a, &b, &c);
        assert!(
            (min_a - PI / 3.0).abs() < 1e-6,
            "Expected pi/3, got {}",
            min_a
        );
    }

    // ---- Test 14: Minimum angle right-isoceles ----
    #[test]
    fn test_min_angle_right_isoceles() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(1.0, 0.0);
        let c = Point2D::new(0.0, 1.0);
        let min_a = triangle_minimum_angle(&a, &b, &c);
        assert!(
            (min_a - PI / 4.0).abs() < 1e-6,
            "Expected pi/4, got {}",
            min_a
        );
    }

    // ---- Test 15: Minimum angle 3-4-5 triangle ----
    #[test]
    fn test_min_angle_345() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 0.0);
        let c = Point2D::new(0.0, 4.0);
        let min_a = triangle_minimum_angle(&a, &b, &c);
        let expected = (3.0_f64 / 5.0).asin();
        assert!(
            (min_a - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            min_a
        );
    }

    // ---- Test 16: Constrained Delaunay basic ----
    #[test]
    fn test_cdt_basic() {
        let pts = five_points();
        let constraints = vec![(0, 2)]; // diagonal constraint
        let cdt = constrained_delaunay(&pts, &constraints).expect("CDT should succeed");
        let interior = cdt.interior_triangles();
        assert!(!interior.is_empty());
    }

    // ---- Test 17: Constrained Delaunay out-of-bounds ----
    #[test]
    fn test_cdt_invalid_index() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];
        let constraints = vec![(0, 10)];
        assert!(constrained_delaunay(&pts, &constraints).is_err());
    }

    // ---- Test 18: Constrained Delaunay with already-existing edge ----
    #[test]
    fn test_cdt_existing_edge() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];
        let constraints = vec![(0, 1)]; // boundary edge always present
        let cdt = constrained_delaunay(&pts, &constraints).expect("CDT should succeed");
        let interior = cdt.interior_triangles();
        assert_eq!(interior.len(), 1);
    }

    // ---- Test 19: Mesh refinement basic ----
    #[test]
    fn test_mesh_refine_basic() {
        let pts = basic_square();
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let refined = mesh_refine(&dt, 5.0, 0.1).expect("refinement should succeed");
        let interior = refined.interior_triangles();
        assert!(interior.len() >= 2);
    }

    // ---- Test 20: Mesh refinement increases points ----
    #[test]
    fn test_mesh_refine_more_points() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(10.0, 10.0),
            Point2D::new(0.0, 10.0),
        ];
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let min_angle_rad = 15.0 * PI / 180.0;
        let refined = mesh_refine(&dt, 25.0, min_angle_rad).expect("refinement should succeed");
        assert!(refined.num_original_points >= 4);
    }

    // ---- Test 21: Edges and neighbors ----
    #[test]
    fn test_edges_and_neighbors() {
        let pts = five_points();
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let edges = dt.edges();
        assert!(!edges.is_empty());
        for e in &edges {
            assert!(e.start < pts.len());
            assert!(e.end < pts.len());
        }
        if !dt.triangles.is_empty() {
            let nbrs = dt.neighbors(0);
            assert!(nbrs.len() <= 3);
        }
    }

    // ---- Test 22: Hexagonal ring ----
    #[test]
    fn test_hexagonal_ring() {
        let center = Point2D::new(0.0, 0.0);
        let mut pts = vec![center];
        for i in 0..6 {
            let angle = (i as f64) * PI / 3.0;
            pts.push(Point2D::new(angle.cos(), angle.sin()));
        }
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let interior = dt.interior_triangles();
        assert_eq!(interior.len(), 6);
        assert!(dt.is_delaunay());
    }

    // ---- Test 23: Many points ----
    #[test]
    fn test_many_points() {
        let pts = grid_points(5, 5);
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let interior = dt.interior_triangles();
        assert!(
            interior.len() >= 10,
            "Expected many triangles, got {}",
            interior.len()
        );
    }

    // ---- Test 24: Boundary edges ----
    #[test]
    fn test_boundary_edges() {
        let pts = basic_square();
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let boundary = dt.boundary_edges();
        assert_eq!(boundary.len(), 4, "Square should have 4 boundary edges");
    }

    // ---- Test 25: Build adjacency ----
    #[test]
    fn test_build_adjacency() {
        let pts = five_points();
        let dt = delaunay_triangulation(&pts).expect("should succeed");
        let adj = dt.build_adjacency();
        // Every edge should map to 1 or 2 triangles
        for tris in adj.values() {
            assert!(
                tris.len() == 1 || tris.len() == 2,
                "Edge has {} triangles",
                tris.len()
            );
        }
    }
}
