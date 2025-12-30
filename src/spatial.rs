//! Spatial Data Structures and Computational Geometry Module
//!
//! This module provides comprehensive spatial algorithms and data structures,
//! built on top of `scirs2-spatial`. It includes:
//!
//! - **KD-Tree**: Efficient spatial indexing for k-nearest neighbor queries
//! - **Convex Hull**: Algorithms for computing convex hulls
//! - **Voronoi Diagrams**: Functions for generating Voronoi diagrams
//! - **Delaunay Triangulation**: Triangulation algorithms
//! - **Utilities**: Helper functions for spatial data processing
//!
//! # Note on Distance Functions
//!
//! This module provides comprehensive distance metrics through scirs2-spatial.
//! For basic distance calculations, you may also use the `distance` module which
//! provides a simpler interface for common distance metrics.
//!
//! # Examples
//!
//! ## KD-Tree for Nearest Neighbor Search
//!
//! ```
//! use numrs2::spatial;
//! use scirs2_core::ndarray::array;
//!
//! // Create some 2D points
//! let points = array![
//!     [2.0, 3.0],
//!     [5.0, 4.0],
//!     [9.0, 6.0],
//!     [4.0, 7.0],
//!     [8.0, 1.0],
//!     [7.0, 2.0]
//! ];
//!
//! // Build a k-d tree
//! let tree = spatial::kdtree::KDTree::new(&points).unwrap();
//!
//! // Query point (as a slice)
//! let query = vec![6.0, 5.0];
//!
//! // Find nearest neighbor
//! let (idx, dist) = tree.query(&query, 1).unwrap();
//! println!("Nearest neighbor index: {}, distance: {}", idx[0], dist[0]);
//!
//! // Find k nearest neighbors
//! let k = 3;
//! let (indices, distances) = tree.query(&query, k).unwrap();
//! println!("k-nearest neighbor indices: {:?}", indices);
//! ```
//!
//! ## Convex Hull Computation
//!
//! ```
//! use numrs2::spatial;
//! use scirs2_core::ndarray::array;
//!
//! // Create 2D points
//! let points = array![
//!     [0.0, 0.0],
//!     [1.0, 0.0],
//!     [0.0, 1.0],
//!     [1.0, 1.0],
//!     [0.5, 0.5],
//! ];
//!
//! // Compute the convex hull (returns hull vertices)
//! let hull = spatial::convex_hull::convex_hull(&points.view()).unwrap();
//!
//! println!("Convex hull vertices: {:?}", hull);
//! ```
//!
//! ## Distance Calculations
//!
//! ```
//! use numrs2::spatial;
//!
//! // Create two points (as slices)
//! let p1 = [1.0_f64, 2.0, 3.0];
//! let p2 = [4.0_f64, 5.0, 6.0];
//!
//! // Calculate various distances (these return T directly, not Result)
//! let euclidean: f64 = spatial::distance::euclidean(&p1, &p2);
//! let manhattan: f64 = spatial::distance::manhattan(&p1, &p2);
//! let chebyshev: f64 = spatial::distance::chebyshev(&p1, &p2);
//!
//! println!("Euclidean distance: {}", euclidean);
//! println!("Manhattan distance: {}", manhattan);
//! println!("Chebyshev distance: {}", chebyshev);
//! ```
//!
//! # KD-Tree Data Structure
//!
//! The KD-tree provides efficient O(log n) nearest neighbor queries on average:
//!
//! - `KDTree::build()`: Build a k-d tree from points
//! - `query()`: Find k nearest neighbors
//! - `query_radius()`: Find all points within a radius
//! - `query_ball_point()`: Query all points within a ball
//! - `query_pairs()`: Query all pairs of points within distance
//!
//! # Convex Hull Algorithms
//!
//! Compute convex hulls for point sets:
//!
//! - `convex_hull()`: Compute convex hull of points
//! - `is_point_in_hull()`: Check if point is in convex hull
//!
//! # Voronoi Diagrams
//!
//! Generate and analyze Voronoi diagrams:
//!
//! - `voronoi_diagram()`: Generate Voronoi diagram
//! - `VoronoiDiagram`: Voronoi diagram structure with vertices and regions
//!
//! # Utilities
//!
//! Additional spatial processing functions:
//!
//! - `delaunay_triangulation()`: Generate Delaunay triangulation
//! - `point_in_polygon()`: Check if point is in polygon
//! - `triangulate()`: Triangulate a polygon
//! - `orient()`: Determine point orientation

// Re-export all scirs2-spatial modules
pub use scirs2_spatial::*;

// Additional NumRS2-specific convenience functions and aliases can be added here

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_kdtree_nearest_neighbor() {
        // Create some 2D points
        let points = array![[2.0, 3.0], [5.0, 4.0], [9.0, 6.0], [4.0, 7.0], [8.0, 1.0],];

        // Build a k-d tree
        let tree = kdtree::KDTree::new(&points).unwrap();

        // Query point
        let query = vec![6.0, 5.0];

        // Find nearest neighbor
        let (idx, dist) = tree.query(&query, 1).unwrap();

        // Should find a neighbor
        assert_eq!(idx.len(), 1);
        assert_eq!(dist.len(), 1);
        assert!(dist[0] >= 0.0);
    }

    #[test]
    fn test_kdtree_k_nearest_neighbors() {
        // Create some 2D points
        let points = array![[2.0, 3.0], [5.0, 4.0], [9.0, 6.0], [4.0, 7.0], [8.0, 1.0],];

        // Build a k-d tree
        let tree = kdtree::KDTree::new(&points).unwrap();

        // Query point
        let query = vec![6.0, 5.0];

        // Find 3 nearest neighbors
        let k = 3;
        let (indices, distances) = tree.query(&query, k).unwrap();

        // Should find exactly 3 neighbors
        assert_eq!(indices.len(), k);
        assert_eq!(distances.len(), k);

        // Distances should be non-negative and sorted
        for i in 0..k {
            assert!(distances[i] >= 0.0);
            if i > 0 {
                assert!(distances[i] >= distances[i - 1]);
            }
        }
    }

    #[test]
    fn test_kdtree_radius_query() {
        // Create some 2D points
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0],];

        // Build a k-d tree
        let tree = kdtree::KDTree::new(&points).unwrap();

        // Query point at origin
        let query = vec![0.0, 0.0];

        // Find all points within radius 1.5
        let radius = 1.5;
        let (indices, distances) = tree.query_radius(&query, radius).unwrap();

        // Should find points at (0,0), (1,0), and (0,1)
        assert!(indices.len() >= 3);

        // All distances should be within radius
        for &d in &distances {
            assert!(d <= radius);
        }
    }

    #[test]
    fn test_distance_euclidean() {
        let p1 = vec![1.0, 2.0, 3.0];
        let p2 = vec![4.0, 5.0, 6.0];

        let dist = distance::euclidean(&p1, &p2);

        // Distance should be sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
        assert!((dist - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_distance_manhattan() {
        let p1 = vec![1.0, 2.0, 3.0];
        let p2 = vec![4.0, 5.0, 6.0];

        let dist = distance::manhattan(&p1, &p2);

        // Distance should be |4-1| + |5-2| + |6-3| = 9
        assert!((dist - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_chebyshev() {
        let p1 = vec![1.0, 2.0, 3.0];
        let p2 = vec![4.0, 5.0, 6.0];

        let dist = distance::chebyshev(&p1, &p2);

        // Distance should be max(|4-1|, |5-2|, |6-3|) = 3
        assert!((dist - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_minkowski() {
        let p1 = vec![1.0, 2.0, 3.0];
        let p2 = vec![4.0, 5.0, 6.0];

        // Minkowski with p=2 should equal Euclidean
        let dist = distance::minkowski(&p1, &p2, 2.0);
        let euclidean_dist = distance::euclidean(&p1, &p2);

        assert!((dist - euclidean_dist).abs() < 1e-10);
    }

    #[test]
    fn test_distance_pdist() {
        // Create 3 points forming a right triangle
        let points = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0],];

        // Calculate pairwise distances using Euclidean metric
        let dists = distance::pdist(&points, distance::euclidean);

        // Should have 3 distances for 3 points: (0,1), (0,2), (1,2)
        assert_eq!(dists.nrows(), 3);
        assert_eq!(dists.ncols(), 3);

        // Distance (0,1) should be 3.0
        assert!((dists[[0, 1]] - 3.0).abs() < 1e-10);

        // Distance (0,2) should be 4.0
        assert!((dists[[0, 2]] - 4.0).abs() < 1e-10);

        // Distance (1,2) should be 5.0 (hypotenuse)
        assert!((dists[[1, 2]] - 5.0).abs() < 1e-10);
    }
}
