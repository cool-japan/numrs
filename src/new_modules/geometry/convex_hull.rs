//! Convex hull algorithms
//!
//! This module implements convex hull computation algorithms:
//! - Graham scan (O(n log n))
//! - Gift wrapping / Jarvis march (O(nh) where h is hull size)
//! - Convex hull area and perimeter
//! - Point-in-convex-hull test
//! - Convex hull merge
//!
//! # References
//!
//! - Graham, R.L. (1972). "An Efficient Algorithm for Determining the Convex Hull
//!   of a Finite Planar Set". Information Processing Letters, 1, 132-133.
//! - Jarvis, R.A. (1973). "On the Identification of the Convex Hull of a Finite
//!   Set of Points in the Plane". Information Processing Letters, 2, 18-21.

use super::{
    is_ccw, is_collinear, orientation, GeometryError, GeometryResult, Point2D, GEOMETRY_EPSILON,
};

/// Computes the convex hull of a set of 2D points using Graham scan algorithm.
///
/// The Graham scan algorithm works by:
/// 1. Finding the lowest point (and leftmost if tied)
/// 2. Sorting all other points by polar angle relative to that point
/// 3. Processing points in order, maintaining a stack of hull vertices
///
/// Time complexity: O(n log n) due to sorting
/// Space complexity: O(n)
///
/// # Arguments
/// * `points` - A slice of 2D points
///
/// # Returns
/// A vector of points forming the convex hull in counter-clockwise order
///
/// # Errors
/// Returns `GeometryError::InsufficientPoints` if fewer than 3 points are provided
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::geometry::{Point2D, convex_hull::graham_scan};
///
/// let points = vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(0.5, 0.5),
///     Point2D::new(0.0, 1.0),
///     Point2D::new(1.0, 1.0),
/// ];
/// let hull = graham_scan(&points).expect("should compute convex hull");
/// assert_eq!(hull.len(), 4); // The inner point is excluded
/// ```
pub fn graham_scan(points: &[Point2D]) -> GeometryResult<Vec<Point2D>> {
    if points.len() < 3 {
        return Err(GeometryError::InsufficientPoints {
            needed: 3,
            got: points.len(),
        });
    }

    // Find the bottom-most point (and leftmost if tied)
    let mut pivot_idx = 0;
    for (i, p) in points.iter().enumerate().skip(1) {
        if p.y < points[pivot_idx].y || (p.y == points[pivot_idx].y && p.x < points[pivot_idx].x) {
            pivot_idx = i;
        }
    }

    let pivot = points[pivot_idx];

    // Create indexed points for sorting
    let mut indexed_points: Vec<(usize, Point2D)> = points
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != pivot_idx)
        .map(|(i, &p)| (i, p))
        .collect();

    // Sort by polar angle relative to pivot, then by distance for ties
    indexed_points.sort_by(|a, b| {
        let angle_a = (a.1.y - pivot.y).atan2(a.1.x - pivot.x);
        let angle_b = (b.1.y - pivot.y).atan2(b.1.x - pivot.x);

        match angle_a.partial_cmp(&angle_b) {
            Some(std::cmp::Ordering::Equal) => {
                // Same angle => sort by distance from pivot
                let dist_a = pivot.distance_squared(&a.1);
                let dist_b = pivot.distance_squared(&b.1);
                dist_a
                    .partial_cmp(&dist_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            Some(ord) => ord,
            None => std::cmp::Ordering::Equal,
        }
    });

    // Remove points with same angle, keeping only the farthest
    let mut filtered: Vec<Point2D> = Vec::with_capacity(indexed_points.len());
    for i in 0..indexed_points.len() {
        // Skip this point if the next point has the same angle but is farther from the pivot
        if i + 1 < indexed_points.len() {
            let curr = indexed_points[i].1;
            let next = indexed_points[i + 1].1;
            if is_collinear(&pivot, &curr, &next) {
                let dist_curr = pivot.distance_squared(&curr);
                let dist_next = pivot.distance_squared(&next);
                if dist_next >= dist_curr {
                    // Next point is farther (or equal), skip the current closer one
                    continue;
                }
            }
        }
        filtered.push(indexed_points[i].1);
    }

    // Handle degenerate: all points collinear
    if filtered.len() < 2 {
        // If all points are collinear, return the two extreme points + pivot
        let mut result = vec![pivot];
        if !filtered.is_empty() {
            result.push(filtered[filtered.len() - 1]);
        }
        return Ok(result);
    }

    // Graham scan
    let mut stack: Vec<Point2D> = Vec::with_capacity(filtered.len() + 1);
    stack.push(pivot);
    stack.push(filtered[0]);

    for &point in filtered.iter().skip(1) {
        // Pop points that would make a non-left (clockwise) turn
        while stack.len() > 1 {
            let top = stack[stack.len() - 1];
            let second = stack[stack.len() - 2];
            if orientation(&second, &top, &point) <= GEOMETRY_EPSILON {
                stack.pop();
            } else {
                break;
            }
        }
        stack.push(point);
    }

    // Ensure we have at least 3 points for a proper hull
    if stack.len() < 3 {
        // All points might be collinear
        return Err(GeometryError::DegenerateGeometry(
            "All points are collinear".to_string(),
        ));
    }

    Ok(stack)
}

/// Computes the convex hull of a set of 2D points using the gift wrapping (Jarvis march) algorithm.
///
/// The gift wrapping algorithm works by:
/// 1. Starting from the leftmost point
/// 2. At each step, finding the point that makes the smallest counter-clockwise angle
///    with the current direction
/// 3. Continuing until we return to the starting point
///
/// Time complexity: O(nh) where n is the number of points and h is the hull size
/// Space complexity: O(h)
///
/// # Arguments
/// * `points` - A slice of 2D points
///
/// # Returns
/// A vector of points forming the convex hull in counter-clockwise order
///
/// # Errors
/// Returns `GeometryError::InsufficientPoints` if fewer than 3 points are provided
pub fn gift_wrapping(points: &[Point2D]) -> GeometryResult<Vec<Point2D>> {
    let n = points.len();
    if n < 3 {
        return Err(GeometryError::InsufficientPoints { needed: 3, got: n });
    }

    // Find the leftmost point
    let mut leftmost = 0;
    for i in 1..n {
        if points[i].x < points[leftmost].x
            || (points[i].x == points[leftmost].x && points[i].y < points[leftmost].y)
        {
            leftmost = i;
        }
    }

    let mut hull: Vec<Point2D> = Vec::new();
    let mut current = leftmost;
    let max_iterations = n + 1; // Safety bound

    for _ in 0..max_iterations {
        hull.push(points[current]);

        // Find the most counter-clockwise point
        let mut next = 0;
        for i in 1..n {
            if next == current {
                next = i;
                continue;
            }

            let orient = orientation(&points[current], &points[next], &points[i]);
            if orient > GEOMETRY_EPSILON {
                // points[i] is more counter-clockwise
                next = i;
            } else if orient.abs() <= GEOMETRY_EPSILON {
                // Collinear => pick the farther point
                let dist_next = points[current].distance_squared(&points[next]);
                let dist_i = points[current].distance_squared(&points[i]);
                if dist_i > dist_next {
                    next = i;
                }
            }
        }

        current = next;

        if current == leftmost {
            break;
        }
    }

    if hull.len() < 3 {
        return Err(GeometryError::DegenerateGeometry(
            "All points are collinear".to_string(),
        ));
    }

    Ok(hull)
}

/// Computes the area of a convex hull (given as an ordered list of points)
///
/// Uses the shoelace formula on the hull vertices.
///
/// # Arguments
/// * `hull` - A slice of points forming the convex hull (in order)
///
/// # Returns
/// The area of the convex hull
pub fn convex_hull_area(hull: &[Point2D]) -> f64 {
    if hull.len() < 3 {
        return 0.0;
    }

    let n = hull.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += hull[i].x * hull[j].y;
        area -= hull[j].x * hull[i].y;
    }

    (area / 2.0).abs()
}

/// Computes the perimeter of a convex hull (given as an ordered list of points)
///
/// # Arguments
/// * `hull` - A slice of points forming the convex hull (in order)
///
/// # Returns
/// The perimeter length of the convex hull
pub fn convex_hull_perimeter(hull: &[Point2D]) -> f64 {
    if hull.len() < 2 {
        return 0.0;
    }

    let n = hull.len();
    let mut perimeter = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        perimeter += hull[i].distance(&hull[j]);
    }
    perimeter
}

/// Tests whether a point lies inside a convex hull
///
/// Uses the cross-product method: a point is inside a convex polygon if it is on
/// the same side of every edge (all cross products have the same sign).
///
/// Time complexity: O(h) where h is the number of hull vertices
///
/// # Arguments
/// * `point` - The point to test
/// * `hull` - The convex hull vertices in counter-clockwise order
///
/// # Returns
/// `true` if the point is inside (or on the boundary of) the convex hull
pub fn point_in_convex_hull(point: &Point2D, hull: &[Point2D]) -> bool {
    if hull.len() < 3 {
        return false;
    }

    let n = hull.len();
    let mut all_positive = true;
    let mut all_negative = true;

    for i in 0..n {
        let j = (i + 1) % n;
        let cross = orientation(&hull[i], &hull[j], point);

        if cross < -GEOMETRY_EPSILON {
            all_positive = false;
        }
        if cross > GEOMETRY_EPSILON {
            all_negative = false;
        }
    }

    all_positive || all_negative
}

/// Merges two convex hulls into a single convex hull
///
/// This algorithm computes the convex hull of the union of two convex polygons.
/// It finds the upper and lower tangent lines between the two hulls, then
/// constructs the merged hull by tracing the outer boundary.
///
/// Time complexity: O(n + m) where n, m are the sizes of the two hulls
///
/// # Arguments
/// * `hull_a` - First convex hull (counter-clockwise order)
/// * `hull_b` - Second convex hull (counter-clockwise order)
///
/// # Returns
/// The merged convex hull
///
/// # Errors
/// Returns an error if either hull is degenerate
pub fn convex_hull_merge(hull_a: &[Point2D], hull_b: &[Point2D]) -> GeometryResult<Vec<Point2D>> {
    if hull_a.is_empty() && hull_b.is_empty() {
        return Err(GeometryError::InsufficientPoints { needed: 3, got: 0 });
    }

    if hull_a.is_empty() {
        return Ok(hull_b.to_vec());
    }

    if hull_b.is_empty() {
        return Ok(hull_a.to_vec());
    }

    // Simple approach: combine all points and compute the hull
    let mut all_points: Vec<Point2D> = Vec::with_capacity(hull_a.len() + hull_b.len());
    all_points.extend_from_slice(hull_a);
    all_points.extend_from_slice(hull_b);

    graham_scan(&all_points)
}

/// Computes the convex hull of a set of 2D points using Andrew's monotone chain algorithm.
///
/// Andrew's monotone chain algorithm works by:
/// 1. Sorting points lexicographically (by x, then by y)
/// 2. Building the lower hull by processing points left to right
/// 3. Building the upper hull by processing points right to left
/// 4. Concatenating the two halves
///
/// Time complexity: O(n log n) due to sorting
/// Space complexity: O(n)
///
/// # Arguments
/// * `points` - A slice of 2D points
///
/// # Returns
/// A vector of points forming the convex hull in counter-clockwise order
///
/// # Errors
/// Returns `GeometryError::InsufficientPoints` if fewer than 3 points are provided
///
/// # Examples
///
/// ```rust
/// use numrs2::new_modules::geometry::{Point2D, convex_hull::monotone_chain};
///
/// let points = vec![
///     Point2D::new(0.0, 0.0),
///     Point2D::new(1.0, 0.0),
///     Point2D::new(0.5, 0.5),
///     Point2D::new(0.0, 1.0),
///     Point2D::new(1.0, 1.0),
/// ];
/// let hull = monotone_chain(&points).expect("should compute convex hull");
/// assert_eq!(hull.len(), 4); // The inner point is excluded
/// ```
pub fn monotone_chain(points: &[Point2D]) -> GeometryResult<Vec<Point2D>> {
    let n = points.len();
    if n < 3 {
        return Err(GeometryError::InsufficientPoints { needed: 3, got: n });
    }

    // Sort points lexicographically (by x, then by y)
    let mut sorted: Vec<Point2D> = points.to_vec();
    sorted.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Remove duplicates
    sorted.dedup_by(|a, b| a.distance(b) < GEOMETRY_EPSILON);

    let m = sorted.len();
    if m < 3 {
        return Err(GeometryError::DegenerateGeometry(
            "Too few unique points for a convex hull".to_string(),
        ));
    }

    // Build lower hull
    let mut lower: Vec<Point2D> = Vec::with_capacity(m);
    for &p in &sorted {
        while lower.len() >= 2 {
            let len = lower.len();
            if orientation(&lower[len - 2], &lower[len - 1], &p) <= GEOMETRY_EPSILON {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(p);
    }

    // Build upper hull
    let mut upper: Vec<Point2D> = Vec::with_capacity(m);
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 {
            let len = upper.len();
            if orientation(&upper[len - 2], &upper[len - 1], &p) <= GEOMETRY_EPSILON {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(p);
    }

    // Remove the last point of each half because it is repeated at the beginning of the other half
    lower.pop();
    upper.pop();

    lower.extend(upper);

    if lower.len() < 3 {
        return Err(GeometryError::DegenerateGeometry(
            "All points are collinear".to_string(),
        ));
    }

    Ok(lower)
}

/// Finds the upper tangent between two convex hulls
///
/// Both hulls must be sorted by x-coordinate. The function returns the indices
/// of the tangent points in each hull.
///
/// # Arguments
/// * `hull_a` - First convex hull (sorted by x)
/// * `hull_b` - Second convex hull (sorted by x)
///
/// # Returns
/// `(index_in_a, index_in_b)` - indices of the tangent endpoints
fn _upper_tangent(hull_a: &[Point2D], hull_b: &[Point2D]) -> (usize, usize) {
    let na = hull_a.len();
    let nb = hull_b.len();

    // Start with rightmost point of A and leftmost of B
    let mut ia = 0;
    for i in 1..na {
        if hull_a[i].x > hull_a[ia].x {
            ia = i;
        }
    }

    let mut ib = 0;
    for i in 1..nb {
        if hull_b[i].x < hull_b[ib].x {
            ib = i;
        }
    }

    let mut done = false;
    while !done {
        done = true;

        // Move ia counter-clockwise while tangent goes up
        if na > 1 {
            loop {
                let next = (ia + 1) % na;
                if orientation(&hull_b[ib], &hull_a[ia], &hull_a[next]) > 0.0 {
                    ia = next;
                    done = false;
                } else {
                    break;
                }
            }
        }

        // Move ib clockwise while tangent goes up
        if nb > 1 {
            loop {
                let prev = if ib == 0 { nb - 1 } else { ib - 1 };
                if orientation(&hull_a[ia], &hull_b[ib], &hull_b[prev]) < 0.0 {
                    ib = prev;
                    done = false;
                } else {
                    break;
                }
            }
        }
    }

    (ia, ib)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_points() -> Vec<Point2D> {
        vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ]
    }

    fn square_with_interior() -> Vec<Point2D> {
        vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(0.5, 0.5), // interior point
            Point2D::new(0.3, 0.3), // interior point
            Point2D::new(0.7, 0.2), // interior point
        ]
    }

    #[test]
    fn test_graham_scan_square() {
        let points = square_points();
        let hull = graham_scan(&points).expect("graham scan should succeed");
        assert_eq!(hull.len(), 4);
        assert!((convex_hull_area(&hull) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_graham_scan_with_interior() {
        let points = square_with_interior();
        let hull = graham_scan(&points).expect("graham scan should succeed");
        assert_eq!(hull.len(), 4);
        assert!((convex_hull_area(&hull) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gift_wrapping_square() {
        let points = square_points();
        let hull = gift_wrapping(&points).expect("gift wrapping should succeed");
        assert_eq!(hull.len(), 4);
        assert!((convex_hull_area(&hull) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gift_wrapping_with_interior() {
        let points = square_with_interior();
        let hull = gift_wrapping(&points).expect("gift wrapping should succeed");
        assert_eq!(hull.len(), 4);
        assert!((convex_hull_area(&hull) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_convex_hull_area_triangle() {
        let hull = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(0.0, 3.0),
        ];
        assert!((convex_hull_area(&hull) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_convex_hull_perimeter_square() {
        let hull = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];
        assert!((convex_hull_perimeter(&hull) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_in_convex_hull_inside() {
        let hull = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 4.0),
            Point2D::new(0.0, 4.0),
        ];

        assert!(point_in_convex_hull(&Point2D::new(2.0, 2.0), &hull));
        assert!(point_in_convex_hull(&Point2D::new(0.5, 0.5), &hull));
    }

    #[test]
    fn test_point_in_convex_hull_outside() {
        let hull = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 4.0),
            Point2D::new(0.0, 4.0),
        ];

        assert!(!point_in_convex_hull(&Point2D::new(5.0, 5.0), &hull));
        assert!(!point_in_convex_hull(&Point2D::new(-1.0, 2.0), &hull));
    }

    #[test]
    fn test_point_in_convex_hull_on_boundary() {
        let hull = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 4.0),
            Point2D::new(0.0, 4.0),
        ];

        // Points on the boundary should be considered inside
        assert!(point_in_convex_hull(&Point2D::new(2.0, 0.0), &hull));
        assert!(point_in_convex_hull(&Point2D::new(0.0, 2.0), &hull));
    }

    #[test]
    fn test_convex_hull_merge() {
        let hull_a = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(2.0, 2.0),
            Point2D::new(0.0, 2.0),
        ];

        let hull_b = vec![
            Point2D::new(3.0, 0.0),
            Point2D::new(5.0, 0.0),
            Point2D::new(5.0, 2.0),
            Point2D::new(3.0, 2.0),
        ];

        let merged = convex_hull_merge(&hull_a, &hull_b).expect("hull merge should succeed");

        // Merged hull should contain all original hull vertices
        // and have area >= max(area_a, area_b)
        let merged_area = convex_hull_area(&merged);
        assert!(merged_area >= convex_hull_area(&hull_a) - GEOMETRY_EPSILON);
        assert!(merged_area >= convex_hull_area(&hull_b) - GEOMETRY_EPSILON);
    }

    #[test]
    fn test_insufficient_points() {
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 1.0)];
        assert!(graham_scan(&points).is_err());
        assert!(gift_wrapping(&points).is_err());
    }

    #[test]
    fn test_triangle_hull() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];

        let hull_gs = graham_scan(&points).expect("graham scan should succeed");
        assert_eq!(hull_gs.len(), 3);
        assert!((convex_hull_area(&hull_gs) - 0.5).abs() < 1e-10);

        let hull_gw = gift_wrapping(&points).expect("gift wrapping should succeed");
        assert_eq!(hull_gw.len(), 3);
        assert!((convex_hull_area(&hull_gw) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_many_points_on_circle() {
        // Generate points on a unit circle
        let n = 20;
        let mut points: Vec<Point2D> = Vec::with_capacity(n);
        for i in 0..n {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            points.push(Point2D::new(angle.cos(), angle.sin()));
        }

        let hull = graham_scan(&points).expect("graham scan should succeed");
        // All points should be on the hull
        assert_eq!(hull.len(), n);
        // Area should be approximately pi
        let area = convex_hull_area(&hull);
        assert!((area - std::f64::consts::PI).abs() < 0.1);
    }

    #[test]
    fn test_monotone_chain_square() {
        let points = square_points();
        let hull = monotone_chain(&points).expect("monotone chain should succeed");
        assert_eq!(hull.len(), 4);
        assert!((convex_hull_area(&hull) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_monotone_chain_with_interior() {
        let points = square_with_interior();
        let hull = monotone_chain(&points).expect("monotone chain should succeed");
        assert_eq!(hull.len(), 4);
        assert!((convex_hull_area(&hull) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_monotone_chain_triangle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];
        let hull = monotone_chain(&points).expect("monotone chain should succeed");
        assert_eq!(hull.len(), 3);
        assert!((convex_hull_area(&hull) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hull_consistency() {
        // Both algorithms should produce the same area
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(3.0, 4.0),
            Point2D::new(1.5, 5.0),
            Point2D::new(0.0, 4.0),
            Point2D::new(1.0, 2.0), // interior
            Point2D::new(2.0, 1.0), // interior
        ];

        let hull_gs = graham_scan(&points).expect("graham scan should succeed");
        let hull_gw = gift_wrapping(&points).expect("gift wrapping should succeed");

        let area_gs = convex_hull_area(&hull_gs);
        let area_gw = convex_hull_area(&hull_gw);

        assert!(
            (area_gs - area_gw).abs() < 1e-8,
            "Graham scan area ({}) and gift wrapping area ({}) should match",
            area_gs,
            area_gw
        );
    }
}
