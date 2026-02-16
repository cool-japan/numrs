//! Polygon operations
//!
//! This module provides comprehensive polygon operations including:
//! - Area computation (shoelace formula)
//! - Centroid computation
//! - Point-in-polygon test (ray casting algorithm)
//! - Polygon convexity check
//! - Polygon simplification (Ramer-Douglas-Peucker algorithm)
//! - Line segment intersection
//! - Polygon clipping (Sutherland-Hodgman algorithm)
//! - Minimum bounding rectangle
//!
//! # References
//!
//! - Meister, A. (1769). "Generalia de genesi figurarum planarum et inde
//!   pendentibus." - Shoelace formula
//! - Douglas, D.; Peucker, T. (1973). "Algorithms for the reduction of the
//!   number of points required to represent a digitized line or its caricature."
//! - Sutherland, I.E.; Hodgman, G.W. (1974). "Reentrant Polygon Clipping."
//!   Communications of the ACM, 17(1), 32-42.

use super::{orientation, GeometryError, GeometryResult, Point2D, GEOMETRY_EPSILON};

/// Computes the signed area of a polygon using the shoelace formula
///
/// The signed area is positive for counter-clockwise vertex ordering
/// and negative for clockwise ordering.
///
/// The shoelace formula computes the area as:
///   A = (1/2) * |sum_{i=0}^{n-1} (x_i * y_{i+1} - x_{i+1} * y_i)|
///
/// # Arguments
/// * `vertices` - A slice of polygon vertices in order
///
/// # Returns
/// The signed area of the polygon
pub fn polygon_signed_area(vertices: &[Point2D]) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }

    let n = vertices.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i].x * vertices[j].y;
        area -= vertices[j].x * vertices[i].y;
    }

    area / 2.0
}

/// Computes the (unsigned) area of a polygon
///
/// # Arguments
/// * `vertices` - A slice of polygon vertices
///
/// # Returns
/// The area of the polygon
pub fn polygon_area(vertices: &[Point2D]) -> f64 {
    polygon_signed_area(vertices).abs()
}

/// Computes the centroid of a polygon
///
/// The centroid is the geometric center of the polygon, computed using:
///   C_x = (1 / 6A) * sum_{i=0}^{n-1} (x_i + x_{i+1}) * (x_i * y_{i+1} - x_{i+1} * y_i)
///   C_y = (1 / 6A) * sum_{i=0}^{n-1} (y_i + y_{i+1}) * (x_i * y_{i+1} - x_{i+1} * y_i)
///
/// # Arguments
/// * `vertices` - A slice of polygon vertices
///
/// # Returns
/// The centroid point, or error if the polygon is degenerate (zero area)
pub fn polygon_centroid(vertices: &[Point2D]) -> GeometryResult<Point2D> {
    if vertices.len() < 3 {
        return Err(GeometryError::InsufficientPoints {
            needed: 3,
            got: vertices.len(),
        });
    }

    let signed_area = polygon_signed_area(vertices);

    if signed_area.abs() < GEOMETRY_EPSILON {
        return Err(GeometryError::DegenerateGeometry(
            "Polygon has zero area, centroid is undefined".to_string(),
        ));
    }

    let n = vertices.len();
    let mut cx = 0.0;
    let mut cy = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        let cross = vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
        cx += (vertices[i].x + vertices[j].x) * cross;
        cy += (vertices[i].y + vertices[j].y) * cross;
    }

    let factor = 1.0 / (6.0 * signed_area);
    Ok(Point2D::new(cx * factor, cy * factor))
}

/// Computes the perimeter of a polygon
///
/// # Arguments
/// * `vertices` - A slice of polygon vertices
///
/// # Returns
/// The perimeter length
pub fn polygon_perimeter(vertices: &[Point2D]) -> f64 {
    if vertices.len() < 2 {
        return 0.0;
    }

    let n = vertices.len();
    let mut perimeter = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        perimeter += vertices[i].distance(&vertices[j]);
    }
    perimeter
}

/// Tests whether a point is inside a polygon using the ray casting algorithm
///
/// Casts a horizontal ray from the test point and counts the number of
/// polygon edge intersections. An odd count means the point is inside.
///
/// # Arguments
/// * `point` - The point to test
/// * `vertices` - The polygon vertices
///
/// # Returns
/// `true` if the point is inside the polygon
pub fn point_in_polygon(point: &Point2D, vertices: &[Point2D]) -> bool {
    if vertices.len() < 3 {
        return false;
    }

    let n = vertices.len();
    let mut inside = false;

    let mut j = n - 1;
    for i in 0..n {
        let vi = &vertices[i];
        let vj = &vertices[j];

        // Check if the ray crosses this edge
        if ((vi.y > point.y) != (vj.y > point.y))
            && (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x)
        {
            inside = !inside;
        }

        j = i;
    }

    inside
}

/// Checks if a polygon is convex
///
/// A polygon is convex if all cross products of consecutive edge vectors have
/// the same sign.
///
/// # Arguments
/// * `vertices` - The polygon vertices
///
/// # Returns
/// `true` if the polygon is convex
pub fn polygon_is_convex(vertices: &[Point2D]) -> bool {
    if vertices.len() < 3 {
        return false;
    }

    let n = vertices.len();
    let mut positive_cross = false;
    let mut negative_cross = false;

    for i in 0..n {
        let a = &vertices[i];
        let b = &vertices[(i + 1) % n];
        let c = &vertices[(i + 2) % n];

        let cross = orientation(a, b, c);

        if cross > GEOMETRY_EPSILON {
            positive_cross = true;
        } else if cross < -GEOMETRY_EPSILON {
            negative_cross = true;
        }

        if positive_cross && negative_cross {
            return false;
        }
    }

    true
}

/// Result type for line segment intersection
#[derive(Debug, Clone)]
pub enum IntersectionResult {
    /// No intersection
    None,
    /// Intersection at a single point
    Point(Point2D),
    /// Segments are collinear and overlap
    Collinear,
}

/// Computes the intersection of two line segments
///
/// Given segments P1-P2 and P3-P4, computes their intersection point if it exists.
///
/// Uses parametric form:
///   P = P1 + t * (P2 - P1) for t in [0, 1]
///   P = P3 + u * (P4 - P3) for u in [0, 1]
///
/// # Arguments
/// * `p1`, `p2` - Endpoints of the first segment
/// * `p3`, `p4` - Endpoints of the second segment
///
/// # Returns
/// An `IntersectionResult` indicating the type and location of intersection
pub fn line_segment_intersection(
    p1: &Point2D,
    p2: &Point2D,
    p3: &Point2D,
    p4: &Point2D,
) -> IntersectionResult {
    let d1 = p2.sub(p1);
    let d2 = p4.sub(p3);

    let denom = d1.x * d2.y - d1.y * d2.x;

    if denom.abs() < GEOMETRY_EPSILON {
        // Lines are parallel. Check if they are collinear.
        let d3 = p3.sub(p1);
        let cross = d3.x * d1.y - d3.y * d1.x;

        if cross.abs() < GEOMETRY_EPSILON {
            // Collinear. Check if segments overlap.
            let t0 = if d1.x.abs() > GEOMETRY_EPSILON {
                (p3.x - p1.x) / d1.x
            } else if d1.y.abs() > GEOMETRY_EPSILON {
                (p3.y - p1.y) / d1.y
            } else {
                return IntersectionResult::None;
            };

            let t1 = if d1.x.abs() > GEOMETRY_EPSILON {
                (p4.x - p1.x) / d1.x
            } else if d1.y.abs() > GEOMETRY_EPSILON {
                (p4.y - p1.y) / d1.y
            } else {
                return IntersectionResult::None;
            };

            let (t_min, t_max) = if t0 < t1 { (t0, t1) } else { (t1, t0) };

            if t_max < -GEOMETRY_EPSILON || t_min > 1.0 + GEOMETRY_EPSILON {
                return IntersectionResult::None;
            }

            return IntersectionResult::Collinear;
        }

        return IntersectionResult::None;
    }

    let d3 = p3.sub(p1);
    let t = (d3.x * d2.y - d3.y * d2.x) / denom;
    let u = (d3.x * d1.y - d3.y * d1.x) / denom;

    if (-GEOMETRY_EPSILON..=1.0 + GEOMETRY_EPSILON).contains(&t)
        && (-GEOMETRY_EPSILON..=1.0 + GEOMETRY_EPSILON).contains(&u)
    {
        let intersection = Point2D::new(p1.x + t * d1.x, p1.y + t * d1.y);
        IntersectionResult::Point(intersection)
    } else {
        IntersectionResult::None
    }
}

/// Simplifies a polygon using the Ramer-Douglas-Peucker algorithm
///
/// This algorithm reduces the number of points in a polygon while maintaining
/// the overall shape within a given tolerance.
///
/// # Algorithm
/// 1. Start with the line segment from first to last point
/// 2. Find the point farthest from the line
/// 3. If the distance is greater than epsilon, keep that point and recurse on both halves
/// 4. Otherwise, discard all intermediate points
///
/// # Arguments
/// * `vertices` - The polygon vertices to simplify
/// * `epsilon` - The maximum allowed deviation from the original polygon
///
/// # Returns
/// A simplified list of vertices
///
/// # Errors
/// Returns an error if epsilon is negative
pub fn polygon_simplify_rdp(vertices: &[Point2D], epsilon: f64) -> GeometryResult<Vec<Point2D>> {
    if epsilon < 0.0 {
        return Err(GeometryError::InvalidPolygon(
            "Epsilon must be non-negative".to_string(),
        ));
    }

    if vertices.len() <= 2 {
        return Ok(vertices.to_vec());
    }

    // For closed polygons, we simplify the open version and close it
    let n = vertices.len();

    // Check if polygon is closed (first == last)
    let is_closed = vertices[0].distance(&vertices[n - 1]) < GEOMETRY_EPSILON;

    if is_closed && n > 3 {
        // Simplify as open polyline, then close
        let open = &vertices[..n - 1];
        let mut simplified = rdp_recursive(open, epsilon);
        // Close the polygon
        if simplified.len() >= 2 {
            simplified.push(simplified[0]);
        }
        return Ok(simplified);
    }

    Ok(rdp_recursive(vertices, epsilon))
}

/// Recursive helper for the Ramer-Douglas-Peucker algorithm
fn rdp_recursive(points: &[Point2D], epsilon: f64) -> Vec<Point2D> {
    if points.len() <= 2 {
        return points.to_vec();
    }

    // Find the point farthest from the line between first and last
    let first = &points[0];
    let last = &points[points.len() - 1];

    let mut max_dist = 0.0;
    let mut max_idx = 0;

    for (i, point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let dist = perpendicular_distance(point, first, last);
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        // Recurse on both halves
        let mut left = rdp_recursive(&points[..=max_idx], epsilon);
        let right = rdp_recursive(&points[max_idx..], epsilon);

        // Merge (remove duplicate at split point)
        left.pop();
        left.extend(right);
        left
    } else {
        // Discard intermediate points
        vec![*first, *last]
    }
}

/// Computes the perpendicular distance from a point to a line segment
fn perpendicular_distance(point: &Point2D, line_start: &Point2D, line_end: &Point2D) -> f64 {
    let dx = line_end.x - line_start.x;
    let dy = line_end.y - line_start.y;

    let line_len_sq = dx * dx + dy * dy;

    if line_len_sq < GEOMETRY_EPSILON * GEOMETRY_EPSILON {
        // Degenerate line segment (start == end)
        return point.distance(line_start);
    }

    let area = ((point.x - line_start.x) * dy - (point.y - line_start.y) * dx).abs();
    area / line_len_sq.sqrt()
}

/// Clips a polygon against a convex clipping polygon using the Sutherland-Hodgman algorithm
///
/// # Algorithm
/// For each edge of the clip polygon:
///   For each edge of the subject polygon:
///     1. If both endpoints are inside, add the second endpoint
///     2. If going from inside to outside, add the intersection
///     3. If going from outside to inside, add the intersection and the second endpoint
///     4. If both are outside, add nothing
///
/// # Arguments
/// * `subject` - The polygon to clip
/// * `clip` - The convex clipping polygon (vertices in counter-clockwise order)
///
/// # Returns
/// The clipped polygon vertices
///
/// # Errors
/// Returns an error if either polygon has fewer than 3 vertices
pub fn polygon_clipping_sutherland_hodgman(
    subject: &[Point2D],
    clip: &[Point2D],
) -> GeometryResult<Vec<Point2D>> {
    if subject.len() < 3 {
        return Err(GeometryError::InsufficientPoints {
            needed: 3,
            got: subject.len(),
        });
    }

    if clip.len() < 3 {
        return Err(GeometryError::InsufficientPoints {
            needed: 3,
            got: clip.len(),
        });
    }

    let mut output = subject.to_vec();

    let clip_n = clip.len();
    for i in 0..clip_n {
        if output.is_empty() {
            break;
        }

        let edge_start = &clip[i];
        let edge_end = &clip[(i + 1) % clip_n];

        let input = output;
        output = Vec::new();

        let n = input.len();
        for j in 0..n {
            let current = &input[j];
            let previous = &input[(j + n - 1) % n];

            let current_inside = is_inside_edge(current, edge_start, edge_end);
            let previous_inside = is_inside_edge(previous, edge_start, edge_end);

            if current_inside {
                if !previous_inside {
                    // Entering the clip region
                    if let Some(intersection) =
                        line_line_intersection(previous, current, edge_start, edge_end)
                    {
                        output.push(intersection);
                    }
                }
                output.push(*current);
            } else if previous_inside {
                // Leaving the clip region
                if let Some(intersection) =
                    line_line_intersection(previous, current, edge_start, edge_end)
                {
                    output.push(intersection);
                }
            }
        }
    }

    Ok(output)
}

/// Checks if a point is on the "inside" of a directed edge
///
/// A point is inside if it is to the left of the edge (counter-clockwise).
fn is_inside_edge(point: &Point2D, edge_start: &Point2D, edge_end: &Point2D) -> bool {
    orientation(edge_start, edge_end, point) >= -GEOMETRY_EPSILON
}

/// Computes the intersection of two infinite lines (defined by two points each)
fn line_line_intersection(
    p1: &Point2D,
    p2: &Point2D,
    p3: &Point2D,
    p4: &Point2D,
) -> Option<Point2D> {
    let d1x = p2.x - p1.x;
    let d1y = p2.y - p1.y;
    let d2x = p4.x - p3.x;
    let d2y = p4.y - p3.y;

    let denom = d1x * d2y - d1y * d2x;

    if denom.abs() < GEOMETRY_EPSILON {
        return None; // Parallel lines
    }

    let t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom;

    Some(Point2D::new(p1.x + t * d1x, p1.y + t * d1y))
}

/// Computes the minimum bounding rectangle (axis-aligned) for a set of points
///
/// Returns the rectangle as `(min_corner, max_corner)`.
///
/// # Arguments
/// * `points` - A slice of 2D points
///
/// # Returns
/// `(bottom_left, top_right)` corners of the bounding rectangle
///
/// # Errors
/// Returns an error if no points are provided
pub fn minimum_bounding_rectangle(points: &[Point2D]) -> GeometryResult<(Point2D, Point2D)> {
    if points.is_empty() {
        return Err(GeometryError::InsufficientPoints { needed: 1, got: 0 });
    }

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

    Ok((Point2D::new(min_x, min_y), Point2D::new(max_x, max_y)))
}

/// Checks if two convex polygons intersect using the Separating Axis Theorem (SAT).
///
/// The SAT states that two convex shapes do not intersect if and only if there exists
/// a separating axis such that the projections of the two shapes onto that axis
/// do not overlap. For convex polygons, it suffices to test axes perpendicular
/// to each edge of both polygons.
///
/// Time complexity: O(n + m) where n, m are the vertex counts
///
/// # Arguments
/// * `poly_a` - Vertices of the first convex polygon
/// * `poly_b` - Vertices of the second convex polygon
///
/// # Returns
/// `true` if the polygons intersect (overlap or touch)
pub fn convex_polygons_intersect(poly_a: &[Point2D], poly_b: &[Point2D]) -> bool {
    if poly_a.len() < 3 || poly_b.len() < 3 {
        return false;
    }

    // Test separating axes from polygon A's edges
    if has_separating_axis(poly_a, poly_b) {
        return false;
    }

    // Test separating axes from polygon B's edges
    if has_separating_axis(poly_b, poly_a) {
        return false;
    }

    true
}

/// Helper: checks if any edge normal of `source` is a separating axis for `source` and `target`.
fn has_separating_axis(source: &[Point2D], target: &[Point2D]) -> bool {
    let n = source.len();
    for i in 0..n {
        let j = (i + 1) % n;

        // Edge vector and its perpendicular (outward normal)
        let edge = source[j].sub(&source[i]);
        let axis = Point2D::new(-edge.y, edge.x);

        // Project all points of both polygons onto the axis
        let (src_min, src_max) = project_polygon(&axis, source);
        let (tgt_min, tgt_max) = project_polygon(&axis, target);

        // Check for gap between projections
        if src_max < tgt_min - GEOMETRY_EPSILON || tgt_max < src_min - GEOMETRY_EPSILON {
            return true;
        }
    }
    false
}

/// Projects a polygon onto an axis and returns (min, max) projection values.
fn project_polygon(axis: &Point2D, vertices: &[Point2D]) -> (f64, f64) {
    let first_proj = axis.dot(&vertices[0]);
    let mut min_proj = first_proj;
    let mut max_proj = first_proj;

    for v in vertices.iter().skip(1) {
        let proj = axis.dot(v);
        if proj < min_proj {
            min_proj = proj;
        }
        if proj > max_proj {
            max_proj = proj;
        }
    }

    (min_proj, max_proj)
}

/// Computes the union of two convex polygons (as the convex hull of their union).
///
/// For convex polygons, this returns the convex hull of all vertices from both polygons.
/// Note that for non-overlapping convex polygons, the result is the convex hull
/// of the combined point set, which may be larger than the true set-theoretic union.
///
/// # Arguments
/// * `poly_a` - Vertices of the first convex polygon
/// * `poly_b` - Vertices of the second convex polygon
///
/// # Returns
/// Vertices of the convex hull of the union
///
/// # Errors
/// Returns an error if the combined point count is insufficient
pub fn convex_polygon_union(
    poly_a: &[Point2D],
    poly_b: &[Point2D],
) -> GeometryResult<Vec<Point2D>> {
    let mut all_points = Vec::with_capacity(poly_a.len() + poly_b.len());
    all_points.extend_from_slice(poly_a);
    all_points.extend_from_slice(poly_b);

    if all_points.len() < 3 {
        return Err(GeometryError::InsufficientPoints {
            needed: 3,
            got: all_points.len(),
        });
    }

    super::convex_hull::graham_scan(&all_points)
}

/// Checks if a polygon is simple (non-self-intersecting).
///
/// A simple polygon has no two non-adjacent edges that intersect.
///
/// Time complexity: O(n^2) brute force
///
/// # Arguments
/// * `vertices` - The polygon vertices
///
/// # Returns
/// `true` if the polygon is simple
pub fn polygon_is_simple(vertices: &[Point2D]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }

    for i in 0..n {
        let i_next = (i + 1) % n;
        for j in (i + 2)..n {
            let j_next = (j + 1) % n;
            // Skip adjacent edges
            if j_next == i {
                continue;
            }
            let result = line_segment_intersection(
                &vertices[i],
                &vertices[i_next],
                &vertices[j],
                &vertices[j_next],
            );
            match result {
                IntersectionResult::Point(_) | IntersectionResult::Collinear => return false,
                IntersectionResult::None => {}
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_area_rectangle() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 3.0),
            Point2D::new(0.0, 3.0),
        ];
        assert!((polygon_area(&vertices) - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_area_triangle() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(2.0, 3.0),
        ];
        assert!((polygon_area(&vertices) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_signed_area_ccw() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        assert!(polygon_signed_area(&vertices) > 0.0);
    }

    #[test]
    fn test_polygon_signed_area_cw() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(1.0, 0.0),
        ];
        assert!(polygon_signed_area(&vertices) < 0.0);
    }

    #[test]
    fn test_polygon_centroid_square() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 4.0),
            Point2D::new(0.0, 4.0),
        ];
        let centroid = polygon_centroid(&vertices).expect("centroid should succeed");
        assert!((centroid.x - 2.0).abs() < 1e-10);
        assert!((centroid.y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_centroid_triangle() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(6.0, 0.0),
            Point2D::new(0.0, 6.0),
        ];
        let centroid = polygon_centroid(&vertices).expect("centroid should succeed");
        assert!((centroid.x - 2.0).abs() < 1e-10);
        assert!((centroid.y - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_perimeter() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(3.0, 4.0),
            Point2D::new(0.0, 4.0),
        ];
        assert!((polygon_perimeter(&vertices) - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_in_polygon_square() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 4.0),
            Point2D::new(0.0, 4.0),
        ];

        // Inside
        assert!(point_in_polygon(&Point2D::new(2.0, 2.0), &vertices));
        assert!(point_in_polygon(&Point2D::new(0.1, 0.1), &vertices));
        assert!(point_in_polygon(&Point2D::new(3.9, 3.9), &vertices));

        // Outside
        assert!(!point_in_polygon(&Point2D::new(5.0, 2.0), &vertices));
        assert!(!point_in_polygon(&Point2D::new(-1.0, 2.0), &vertices));
        assert!(!point_in_polygon(&Point2D::new(2.0, -1.0), &vertices));
        assert!(!point_in_polygon(&Point2D::new(2.0, 5.0), &vertices));
    }

    #[test]
    fn test_point_in_polygon_triangle() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(2.0, 4.0),
        ];

        assert!(point_in_polygon(&Point2D::new(2.0, 1.0), &vertices));
        assert!(!point_in_polygon(&Point2D::new(0.0, 4.0), &vertices));
    }

    #[test]
    fn test_polygon_is_convex() {
        // Convex square
        let square = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        assert!(polygon_is_convex(&square));

        // Non-convex (L-shape)
        let l_shape = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(2.0, 1.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(1.0, 2.0),
            Point2D::new(0.0, 2.0),
        ];
        assert!(!polygon_is_convex(&l_shape));
    }

    #[test]
    fn test_line_segment_intersection_crossing() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(2.0, 2.0);
        let p3 = Point2D::new(0.0, 2.0);
        let p4 = Point2D::new(2.0, 0.0);

        match line_segment_intersection(&p1, &p2, &p3, &p4) {
            IntersectionResult::Point(p) => {
                assert!((p.x - 1.0).abs() < 1e-10);
                assert!((p.y - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected point intersection"),
        }
    }

    #[test]
    fn test_line_segment_intersection_no_crossing() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(1.0, 0.0);
        let p3 = Point2D::new(2.0, 1.0);
        let p4 = Point2D::new(3.0, 1.0);

        match line_segment_intersection(&p1, &p2, &p3, &p4) {
            IntersectionResult::None => {}
            _ => panic!("Expected no intersection"),
        }
    }

    #[test]
    fn test_line_segment_intersection_parallel() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(1.0, 0.0);
        let p3 = Point2D::new(0.0, 1.0);
        let p4 = Point2D::new(1.0, 1.0);

        match line_segment_intersection(&p1, &p2, &p3, &p4) {
            IntersectionResult::None => {}
            _ => panic!("Expected no intersection for parallel segments"),
        }
    }

    #[test]
    fn test_line_segment_intersection_collinear_overlapping() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(2.0, 0.0);
        let p3 = Point2D::new(1.0, 0.0);
        let p4 = Point2D::new(3.0, 0.0);

        match line_segment_intersection(&p1, &p2, &p3, &p4) {
            IntersectionResult::Collinear => {}
            _ => panic!("Expected collinear intersection"),
        }
    }

    #[test]
    fn test_polygon_simplify_rdp_simple() {
        // A triangle should not be simplified
        let triangle = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];
        let simplified =
            polygon_simplify_rdp(&triangle, 0.1).expect("simplification should succeed");
        assert!(simplified.len() <= 3);
    }

    #[test]
    fn test_polygon_simplify_rdp_collinear() {
        // Points along a line should simplify to two endpoints
        let line = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(4.0, 0.0),
        ];
        let simplified = polygon_simplify_rdp(&line, 0.1).expect("simplification should succeed");
        assert_eq!(simplified.len(), 2);
        assert!((simplified[0].x - 0.0).abs() < 1e-10);
        assert!((simplified[1].x - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_simplify_rdp_nearly_straight() {
        // Slightly off-line points with small epsilon
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.01),
            Point2D::new(2.0, -0.01),
            Point2D::new(3.0, 0.02),
            Point2D::new(4.0, 0.0),
        ];

        // Large epsilon: simplify to line
        let simplified = polygon_simplify_rdp(&points, 0.1).expect("simplification should succeed");
        assert_eq!(simplified.len(), 2);

        // Small epsilon: keep more detail
        let detailed = polygon_simplify_rdp(&points, 0.005).expect("simplification should succeed");
        assert!(detailed.len() > 2);
    }

    #[test]
    fn test_polygon_simplify_rdp_negative_epsilon() {
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0)];
        assert!(polygon_simplify_rdp(&points, -0.1).is_err());
    }

    #[test]
    fn test_polygon_clipping_sutherland_hodgman() {
        // Clip a diamond against a square
        let subject = vec![
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 1.0),
            Point2D::new(1.0, 2.0),
            Point2D::new(0.0, 1.0),
        ];

        let clip = vec![
            Point2D::new(0.5, 0.0),
            Point2D::new(1.5, 0.0),
            Point2D::new(1.5, 2.0),
            Point2D::new(0.5, 2.0),
        ];

        let result =
            polygon_clipping_sutherland_hodgman(&subject, &clip).expect("clipping should succeed");

        // The result should be a non-empty polygon
        assert!(result.len() >= 3);

        // The area of the clipped polygon should be less than or equal to both inputs
        let clipped_area = polygon_area(&result);
        let subject_area = polygon_area(&subject);
        let clip_area = polygon_area(&clip);

        assert!(clipped_area <= subject_area + GEOMETRY_EPSILON);
        assert!(clipped_area <= clip_area + GEOMETRY_EPSILON);
    }

    #[test]
    fn test_polygon_clipping_fully_inside() {
        // Small square inside a big square
        let subject = vec![
            Point2D::new(1.0, 1.0),
            Point2D::new(2.0, 1.0),
            Point2D::new(2.0, 2.0),
            Point2D::new(1.0, 2.0),
        ];

        let clip = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(3.0, 3.0),
            Point2D::new(0.0, 3.0),
        ];

        let result =
            polygon_clipping_sutherland_hodgman(&subject, &clip).expect("clipping should succeed");

        assert_eq!(result.len(), 4);
        let clipped_area = polygon_area(&result);
        assert!((clipped_area - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_minimum_bounding_rectangle() {
        let points = vec![
            Point2D::new(1.0, 2.0),
            Point2D::new(3.0, 5.0),
            Point2D::new(-1.0, 0.0),
            Point2D::new(4.0, 3.0),
        ];

        let (min_pt, max_pt) =
            minimum_bounding_rectangle(&points).expect("bounding rect should succeed");

        assert!((min_pt.x - (-1.0)).abs() < 1e-10);
        assert!((min_pt.y - 0.0).abs() < 1e-10);
        assert!((max_pt.x - 4.0).abs() < 1e-10);
        assert!((max_pt.y - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_minimum_bounding_rectangle_single_point() {
        let points = vec![Point2D::new(3.0, 4.0)];

        let (min_pt, max_pt) =
            minimum_bounding_rectangle(&points).expect("bounding rect should succeed");

        assert!((min_pt.x - 3.0).abs() < 1e-10);
        assert!((min_pt.y - 4.0).abs() < 1e-10);
        assert!((max_pt.x - 3.0).abs() < 1e-10);
        assert!((max_pt.y - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_minimum_bounding_rectangle_empty() {
        let points: Vec<Point2D> = vec![];
        assert!(minimum_bounding_rectangle(&points).is_err());
    }

    #[test]
    fn test_polygon_area_irregular() {
        // Pentagon with known area
        // Regular pentagon with circumradius 1 centered at origin
        let n = 5;
        let mut vertices = Vec::with_capacity(n);
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            vertices.push(Point2D::new(angle.cos(), angle.sin()));
        }

        let area = polygon_area(&vertices);
        // Area of regular pentagon with circumradius 1 = (5/2) * sin(2*pi/5)
        let expected = 2.5 * (2.0 * std::f64::consts::PI / 5.0).sin();
        assert!((area - expected).abs() < 1e-8);
    }

    #[test]
    fn test_perpendicular_distance() {
        let point = Point2D::new(1.0, 1.0);
        let start = Point2D::new(0.0, 0.0);
        let end = Point2D::new(2.0, 0.0);

        let dist = perpendicular_distance(&point, &start, &end);
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_centroid_degenerate() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
        ];
        assert!(polygon_centroid(&vertices).is_err());
    }

    #[test]
    fn test_convex_polygons_intersect_overlapping() {
        let sq1 = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(2.0, 2.0),
            Point2D::new(0.0, 2.0),
        ];
        let sq2 = vec![
            Point2D::new(1.0, 1.0),
            Point2D::new(3.0, 1.0),
            Point2D::new(3.0, 3.0),
            Point2D::new(1.0, 3.0),
        ];
        assert!(convex_polygons_intersect(&sq1, &sq2));
    }

    #[test]
    fn test_convex_polygons_intersect_separated() {
        let sq1 = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        let sq2 = vec![
            Point2D::new(5.0, 5.0),
            Point2D::new(6.0, 5.0),
            Point2D::new(6.0, 6.0),
            Point2D::new(5.0, 6.0),
        ];
        assert!(!convex_polygons_intersect(&sq1, &sq2));
    }

    #[test]
    fn test_convex_polygon_union() {
        let sq1 = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        let sq2 = vec![
            Point2D::new(0.5, 0.0),
            Point2D::new(1.5, 0.0),
            Point2D::new(1.5, 1.0),
            Point2D::new(0.5, 1.0),
        ];
        let union_hull = convex_polygon_union(&sq1, &sq2).expect("union should succeed");
        let area = polygon_area(&union_hull);
        // Union hull: [0,0] [1.5,0] [1.5,1] [0,1] => area 1.5
        assert!((area - 1.5).abs() < 1e-8);
    }

    #[test]
    fn test_polygon_is_simple_square() {
        let sq = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        assert!(polygon_is_simple(&sq));
    }

    #[test]
    fn test_polygon_is_simple_bowtie() {
        // Self-intersecting bowtie
        let bowtie = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 2.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(0.0, 2.0),
        ];
        assert!(!polygon_is_simple(&bowtie));
    }
}
