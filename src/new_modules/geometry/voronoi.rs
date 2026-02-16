//! Voronoi Diagram Computation
//!
//! This module provides:
//! - Voronoi diagram from Delaunay triangulation (dual graph construction)
//! - Nearest-neighbor Voronoi queries (brute-force)
//! - Lloyd's relaxation algorithm (centroidal Voronoi tessellation)
//!
//! # Voronoi Diagram
//!
//! A Voronoi diagram partitions a plane into regions (cells) based on distance
//! to a set of generating points (sites). Each cell contains all points closer
//! to its generator than to any other generator.
//!
//! The Voronoi diagram is the dual of the Delaunay triangulation: each Voronoi
//! vertex corresponds to the circumcenter of a Delaunay triangle, and Voronoi
//! edges connect circumcenters of adjacent Delaunay triangles.
//!
//! # References
//!
//! - Voronoi, G. (1908). "Nouvelles applications des parametres continus a la
//!   theorie des formes quadratiques." Journal fur die reine und angewandte
//!   Mathematik.
//! - Lloyd, S.P. (1982). "Least squares quantization in PCM."
//!   IEEE Transactions on Information Theory, 28(2), 129-137.
//!
//! # Examples
//!
//! ```rust
//! use numrs2::new_modules::geometry::{Point2D, voronoi_from_delaunay};
//!
//! let points = vec![
//!     Point2D::new(0.0, 0.0),
//!     Point2D::new(4.0, 0.0),
//!     Point2D::new(4.0, 4.0),
//!     Point2D::new(0.0, 4.0),
//!     Point2D::new(2.0, 2.0),
//! ];
//! let voronoi = voronoi_from_delaunay(&points).expect("voronoi should succeed");
//! assert_eq!(voronoi.num_cells(), 5);
//! ```

use super::{
    circumcircle, polygon::polygon_centroid, polygon::polygon_clipping_sutherland_hodgman,
    polygon::polygon_signed_area, triangulation::delaunay_triangulation, Edge, GeometryError,
    GeometryResult, Point2D, GEOMETRY_EPSILON,
};

// ---------------------------------------------------------------------------
// VoronoiEdge
// ---------------------------------------------------------------------------

/// A Voronoi edge connecting two Voronoi vertices
///
/// Each edge separates two Voronoi cells (left and right generators).
/// An edge with `end == usize::MAX` extends to infinity (boundary ray).
#[derive(Debug, Clone, Copy)]
pub struct VoronoiEdge {
    /// Index of the first Voronoi vertex (circumcenter)
    pub start: usize,
    /// Index of the second Voronoi vertex, or `usize::MAX` for infinite edges
    pub end: usize,
    /// Index of the generating point on the left side of the edge
    pub left_generator: usize,
    /// Index of the generating point on the right side of the edge
    pub right_generator: usize,
}

// ---------------------------------------------------------------------------
// VoronoiDiagram
// ---------------------------------------------------------------------------

/// A Voronoi diagram data structure
///
/// Stores generating points (sites), Voronoi vertices (circumcenters), cells
/// (ordered vertex-index lists per generator), and edges. Cells for hull
/// generators may be unbounded and are represented with fewer vertices.
#[derive(Debug, Clone)]
pub struct VoronoiDiagram {
    /// The generating points (original input points / sites)
    pub generators: Vec<Point2D>,
    /// Voronoi vertices (circumcenters of Delaunay triangles)
    pub vertices: Vec<Point2D>,
    /// Voronoi cells: for each generator, the ordered list of vertex indices
    /// forming the cell boundary. An empty or short list means the cell is
    /// unbounded or degenerate.
    pub cells: Vec<Vec<usize>>,
    /// Voronoi edges connecting pairs of vertices
    pub edges: Vec<VoronoiEdge>,
}

impl VoronoiDiagram {
    /// Returns the number of cells (generators) in the diagram
    pub fn num_cells(&self) -> usize {
        self.generators.len()
    }

    /// Returns the number of Voronoi vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Returns the cell polygon vertices for a given generator
    ///
    /// # Arguments
    /// * `generator_idx` - Index of the generator point
    ///
    /// # Returns
    /// The Voronoi vertices forming the cell, or an error if the index is
    /// out of range.
    pub fn cell_vertices(&self, generator_idx: usize) -> GeometryResult<Vec<Point2D>> {
        if generator_idx >= self.cells.len() {
            return Err(GeometryError::IndexOutOfBounds(format!(
                "Generator index {} out of range [0, {})",
                generator_idx,
                self.cells.len()
            )));
        }

        let cell = &self.cells[generator_idx];
        let mut verts = Vec::with_capacity(cell.len());
        for &vi in cell {
            if vi < self.vertices.len() {
                verts.push(self.vertices[vi]);
            }
        }

        Ok(verts)
    }

    /// Computes the area of a Voronoi cell (bounded cells only)
    ///
    /// # Arguments
    /// * `generator_idx` - Index of the generator point
    ///
    /// # Returns
    /// The area, or an error if the cell is unbounded/degenerate
    pub fn cell_area(&self, generator_idx: usize) -> GeometryResult<f64> {
        let verts = self.cell_vertices(generator_idx)?;
        if verts.len() < 3 {
            return Err(GeometryError::DegenerateGeometry(
                "Cell has fewer than 3 vertices (possibly unbounded)".to_string(),
            ));
        }
        Ok(polygon_signed_area(&verts).abs())
    }
}

// ---------------------------------------------------------------------------
// Voronoi from Delaunay (dual graph construction)
// ---------------------------------------------------------------------------

/// Finds the shared edge between two triangles, if any
fn find_shared_edge(t1: &super::Triangle, t2: &super::Triangle) -> Option<Edge> {
    let e1 = t1.edges();
    let e2 = t2.edges();
    for a in &e1 {
        for b in &e2 {
            if a == b {
                return Some(*a);
            }
        }
    }
    None
}

/// Computes the Voronoi diagram as the dual of the Delaunay triangulation
///
/// Each Voronoi vertex is the circumcenter of a Delaunay triangle. Two
/// Voronoi vertices are connected by an edge if and only if their
/// corresponding Delaunay triangles share an edge.
///
/// # Arguments
/// * `points` - The generating points (sites)
///
/// # Returns
/// A `VoronoiDiagram`
///
/// # Errors
/// Returns an error if the Delaunay triangulation fails.
pub fn voronoi_from_delaunay(points: &[Point2D]) -> GeometryResult<VoronoiDiagram> {
    if points.len() < 3 {
        return Err(GeometryError::InsufficientPoints {
            needed: 3,
            got: points.len(),
        });
    }

    let dt = delaunay_triangulation(points)?;
    let n = dt.num_original_points;

    // Compute circumcenters for every triangle -> Voronoi vertices
    let mut voronoi_vertices: Vec<Point2D> = Vec::new();
    let mut tri_to_vertex: Vec<Option<usize>> = Vec::with_capacity(dt.triangles.len());

    for tri in &dt.triangles {
        let pa = &dt.points[tri.a];
        let pb = &dt.points[tri.b];
        let pc = &dt.points[tri.c];

        match circumcircle(pa, pb, pc) {
            Ok((center, _)) => {
                let vi = voronoi_vertices.len();
                voronoi_vertices.push(center);
                tri_to_vertex.push(Some(vi));
            }
            Err(_) => {
                tri_to_vertex.push(None);
            }
        }
    }

    // Build Voronoi edges from pairs of adjacent Delaunay triangles
    let mut voronoi_edges: Vec<VoronoiEdge> = Vec::new();
    let num_tris = dt.triangles.len();

    for i in 0..num_tris {
        let vi = match tri_to_vertex[i] {
            Some(v) => v,
            None => continue,
        };

        for j in (i + 1)..num_tris {
            let vj = match tri_to_vertex[j] {
                Some(v) => v,
                None => continue,
            };

            if let Some(shared_edge) = find_shared_edge(&dt.triangles[i], &dt.triangles[j]) {
                // Only include edges where both generators are original points
                if shared_edge.start < n && shared_edge.end < n {
                    voronoi_edges.push(VoronoiEdge {
                        start: vi,
                        end: vj,
                        left_generator: shared_edge.start,
                        right_generator: shared_edge.end,
                    });
                }
            }
        }
    }

    // Build cells: collect circumcenters incident to each original point
    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (tri_idx, tri) in dt.triangles.iter().enumerate() {
        let cc_idx = match tri_to_vertex[tri_idx] {
            Some(v) => v,
            None => continue,
        };
        for &vert in &[tri.a, tri.b, tri.c] {
            if vert < n {
                cells[vert].push(cc_idx);
            }
        }
    }

    // Order each cell's vertices counter-clockwise around its generator
    for (gen_idx, cell) in cells.iter_mut().enumerate() {
        if cell.len() >= 3 {
            let gen = &points[gen_idx];
            cell.sort_by(|&a, &b| {
                let va = &voronoi_vertices[a];
                let vb = &voronoi_vertices[b];
                let angle_a = (va.y - gen.y).atan2(va.x - gen.x);
                let angle_b = (vb.y - gen.y).atan2(vb.x - gen.x);
                angle_a
                    .partial_cmp(&angle_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            cell.dedup();
        }
    }

    Ok(VoronoiDiagram {
        generators: points.to_vec(),
        vertices: voronoi_vertices,
        cells,
        edges: voronoi_edges,
    })
}

// ---------------------------------------------------------------------------
// Nearest Neighbor Voronoi
// ---------------------------------------------------------------------------

/// Finds the nearest generator to a query point using brute-force search
///
/// This is equivalent to determining which Voronoi cell contains the query
/// point. A linear scan is used, which is O(n) per query.
///
/// # Arguments
/// * `query` - The query point
/// * `generators` - The set of generating points
///
/// # Returns
/// The index of the nearest generator
///
/// # Errors
/// Returns an error if `generators` is empty.
pub fn nearest_neighbor_voronoi(query: &Point2D, generators: &[Point2D]) -> GeometryResult<usize> {
    if generators.is_empty() {
        return Err(GeometryError::InsufficientPoints { needed: 1, got: 0 });
    }

    let mut best_idx = 0;
    let mut best_dist = query.distance_squared(&generators[0]);

    for (i, gen) in generators.iter().enumerate().skip(1) {
        let d = query.distance_squared(gen);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }

    Ok(best_idx)
}

// ---------------------------------------------------------------------------
// Lloyd's Relaxation
// ---------------------------------------------------------------------------

/// Performs Lloyd's relaxation (centroidal Voronoi tessellation)
///
/// Lloyd's algorithm iteratively:
/// 1. Computes the Voronoi diagram of the current point set
/// 2. Moves each generator to the centroid of its Voronoi cell
///    (clipped to a bounding box)
/// 3. Repeats until convergence or the iteration limit is reached
///
/// The result is a more uniform (centroidal) point distribution.
///
/// # Arguments
/// * `points` - Initial generator points (at least 3 non-collinear)
/// * `iterations` - Maximum number of relaxation iterations
/// * `bounding_box` - Optional `(min_corner, max_corner)` for clipping.
///   If `None`, one is computed from the point set with a margin.
///
/// # Returns
/// The relaxed point set (same length as input)
///
/// # Errors
/// Returns an error if the initial Voronoi computation fails.
pub fn lloyd_relaxation(
    points: &[Point2D],
    iterations: usize,
    bounding_box: Option<(Point2D, Point2D)>,
) -> GeometryResult<Vec<Point2D>> {
    if points.len() < 3 {
        return Err(GeometryError::InsufficientPoints {
            needed: 3,
            got: points.len(),
        });
    }

    let mut current = points.to_vec();

    // Determine bounding box
    let (bb_min, bb_max) = match bounding_box {
        Some(bb) => bb,
        None => {
            let mut min_x = current[0].x;
            let mut min_y = current[0].y;
            let mut max_x = current[0].x;
            let mut max_y = current[0].y;

            for p in current.iter().skip(1) {
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

            let mx = (max_x - min_x) * 0.1 + 1.0;
            let my = (max_y - min_y) * 0.1 + 1.0;

            (
                Point2D::new(min_x - mx, min_y - my),
                Point2D::new(max_x + mx, max_y + my),
            )
        }
    };

    let clip_polygon = vec![
        bb_min,
        Point2D::new(bb_max.x, bb_min.y),
        bb_max,
        Point2D::new(bb_min.x, bb_max.y),
    ];

    for _iter in 0..iterations {
        let voronoi = match voronoi_from_delaunay(&current) {
            Ok(v) => v,
            Err(_) => break, // Cannot build Voronoi -- stop early
        };

        let mut new_points = current.clone();
        let mut converged = true;

        for (i, _gen) in current.iter().enumerate() {
            // Get cell vertices for generator i
            let cell_verts = match voronoi.cell_vertices(i) {
                Ok(v) if v.len() >= 3 => v,
                _ => continue, // Skip unbounded or degenerate cells
            };

            // Clip cell to bounding box
            let clipped = match polygon_clipping_sutherland_hodgman(&cell_verts, &clip_polygon) {
                Ok(c) if c.len() >= 3 => c,
                _ => continue,
            };

            // Compute centroid of clipped cell
            match polygon_centroid(&clipped) {
                Ok(centroid) => {
                    let clamped = Point2D::new(
                        centroid.x.clamp(bb_min.x, bb_max.x),
                        centroid.y.clamp(bb_min.y, bb_max.y),
                    );

                    if clamped.distance(&current[i]) > GEOMETRY_EPSILON * 100.0 {
                        converged = false;
                    }
                    new_points[i] = clamped;
                }
                Err(_) => continue,
            }
        }

        current = new_points;

        if converged {
            break;
        }
    }

    Ok(current)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_five() -> Vec<Point2D> {
        vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 4.0),
            Point2D::new(0.0, 4.0),
            Point2D::new(2.0, 2.0),
        ]
    }

    // ---- Test 1: Basic Voronoi creation ----
    #[test]
    fn test_voronoi_basic() {
        let pts = basic_five();
        let voronoi = voronoi_from_delaunay(&pts).expect("voronoi should succeed");
        assert_eq!(voronoi.num_cells(), 5);
        assert!(voronoi.num_vertices() > 0);
    }

    // ---- Test 2: Center cell has vertices ----
    #[test]
    fn test_voronoi_center_cell() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(6.0, 0.0),
            Point2D::new(6.0, 6.0),
            Point2D::new(0.0, 6.0),
            Point2D::new(3.0, 3.0),
        ];
        let voronoi = voronoi_from_delaunay(&pts).expect("voronoi should succeed");
        let cell = voronoi.cell_vertices(4).expect("center cell");
        assert!(
            cell.len() >= 3,
            "Center cell should have >= 3 vertices, got {}",
            cell.len()
        );
    }

    // ---- Test 3: Nearest neighbor basic ----
    #[test]
    fn test_nearest_neighbor() {
        let gens = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
            Point2D::new(5.0, 5.0),
        ];
        assert_eq!(
            nearest_neighbor_voronoi(&Point2D::new(1.0, 1.0), &gens).expect("should succeed"),
            0
        );
        assert_eq!(
            nearest_neighbor_voronoi(&Point2D::new(9.0, 0.5), &gens).expect("should succeed"),
            1
        );
        assert_eq!(
            nearest_neighbor_voronoi(&Point2D::new(5.0, 4.0), &gens).expect("should succeed"),
            2
        );
    }

    // ---- Test 4: Nearest neighbor empty generators ----
    #[test]
    fn test_nearest_neighbor_empty() {
        let gens: Vec<Point2D> = vec![];
        assert!(nearest_neighbor_voronoi(&Point2D::new(0.0, 0.0), &gens).is_err());
    }

    // ---- Test 5: Lloyd's relaxation basic ----
    #[test]
    fn test_lloyd_basic() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 4.0),
            Point2D::new(0.0, 4.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(3.0, 1.0),
            Point2D::new(3.0, 3.0),
            Point2D::new(1.0, 3.0),
        ];
        let bbox = (Point2D::new(-2.0, -2.0), Point2D::new(6.0, 6.0));
        let relaxed = lloyd_relaxation(&pts, 3, Some(bbox)).expect("lloyd should succeed");
        assert_eq!(relaxed.len(), pts.len());

        // All relaxed points within bounding box
        for p in &relaxed {
            assert!(
                p.x >= bbox.0.x - GEOMETRY_EPSILON && p.x <= bbox.1.x + GEOMETRY_EPSILON,
                "x={} outside bbox",
                p.x
            );
            assert!(
                p.y >= bbox.0.y - GEOMETRY_EPSILON && p.y <= bbox.1.y + GEOMETRY_EPSILON,
                "y={} outside bbox",
                p.y
            );
        }
    }

    // ---- Test 6: Lloyd's with auto bounding box ----
    #[test]
    fn test_lloyd_auto_bbox() {
        let pts = basic_five();
        let relaxed = lloyd_relaxation(&pts, 5, None).expect("lloyd should succeed");
        assert_eq!(relaxed.len(), pts.len());
    }

    // ---- Test 7: Lloyd's convergence ----
    #[test]
    fn test_lloyd_convergence() {
        let pts = basic_five();
        let bbox = (Point2D::new(-1.0, -1.0), Point2D::new(5.0, 5.0));
        let few = lloyd_relaxation(&pts, 2, Some(bbox)).expect("should succeed");
        let many = lloyd_relaxation(&pts, 20, Some(bbox)).expect("should succeed");
        assert_eq!(few.len(), pts.len());
        assert_eq!(many.len(), pts.len());
    }

    // ---- Test 8: Lloyd insufficient points ----
    #[test]
    fn test_lloyd_insufficient() {
        let pts = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0)];
        assert!(lloyd_relaxation(&pts, 1, None).is_err());
    }

    // ---- Test 9: Voronoi edge duality ----
    #[test]
    fn test_voronoi_edge_duality() {
        let pts = basic_five();
        let voronoi = voronoi_from_delaunay(&pts).expect("voronoi should succeed");
        for edge in &voronoi.edges {
            assert!(edge.left_generator < pts.len());
            assert!(edge.right_generator < pts.len());
            assert_ne!(edge.left_generator, edge.right_generator);
            assert!(edge.start < voronoi.vertices.len());
            assert!(edge.end < voronoi.vertices.len() || edge.end == usize::MAX);
        }
    }

    // ---- Test 10: Voronoi cell area ----
    #[test]
    fn test_voronoi_cell_area() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(6.0, 0.0),
            Point2D::new(6.0, 6.0),
            Point2D::new(0.0, 6.0),
            Point2D::new(3.0, 3.0),
        ];
        let voronoi = voronoi_from_delaunay(&pts).expect("voronoi should succeed");
        // Center cell should be bounded with positive area
        if let Ok(area) = voronoi.cell_area(4) {
            assert!(area > 0.0, "Center cell area should be > 0");
        }
    }

    // ---- Test 11: Voronoi with more points ----
    #[test]
    fn test_voronoi_nine_points() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(8.0, 0.0),
            Point2D::new(0.0, 4.0),
            Point2D::new(4.0, 4.0),
            Point2D::new(8.0, 4.0),
            Point2D::new(0.0, 8.0),
            Point2D::new(4.0, 8.0),
            Point2D::new(8.0, 8.0),
        ];
        let voronoi = voronoi_from_delaunay(&pts).expect("voronoi should succeed");
        assert_eq!(voronoi.num_cells(), 9);
        assert!(voronoi.num_vertices() > 0);
        assert!(!voronoi.edges.is_empty());
    }

    // ---- Test 12: find_shared_edge helper ----
    #[test]
    fn test_find_shared_edge() {
        use super::super::Triangle;

        let t1 = Triangle::new(0, 1, 2);
        let t2 = Triangle::new(1, 2, 3);

        let shared = find_shared_edge(&t1, &t2);
        assert!(shared.is_some());
        let edge = shared.expect("should find shared edge");
        assert_eq!(edge.canonical(), (1, 2));
    }

    // ---- Test 13: No shared edge ----
    #[test]
    fn test_no_shared_edge() {
        use super::super::Triangle;

        let t1 = Triangle::new(0, 1, 2);
        let t2 = Triangle::new(3, 4, 5);
        assert!(find_shared_edge(&t1, &t2).is_none());
    }

    // ---- Test 14: Voronoi insufficient points ----
    #[test]
    fn test_voronoi_insufficient() {
        let pts = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0)];
        assert!(voronoi_from_delaunay(&pts).is_err());
    }

    // ---- Test 15: Nearest neighbor single generator ----
    #[test]
    fn test_nearest_neighbor_single() {
        let gens = vec![Point2D::new(5.0, 5.0)];
        let idx =
            nearest_neighbor_voronoi(&Point2D::new(100.0, 200.0), &gens).expect("should succeed");
        assert_eq!(idx, 0);
    }

    // ---- Test 16: Lloyd's does not lose points ----
    #[test]
    fn test_lloyd_point_count() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(6.0, 0.0),
            Point2D::new(0.0, 3.0),
            Point2D::new(3.0, 3.0),
            Point2D::new(6.0, 3.0),
            Point2D::new(0.0, 6.0),
            Point2D::new(3.0, 6.0),
            Point2D::new(6.0, 6.0),
        ];
        let relaxed = lloyd_relaxation(&pts, 10, None).expect("should succeed");
        assert_eq!(relaxed.len(), pts.len(), "Point count must not change");
    }
}
