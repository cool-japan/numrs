//! Helper functions for multi-objective test problems
//!
//! This module provides shared utility functions used by ZDT and DTLZ benchmark
//! problem implementations.

use num_traits::Float;

/// Compute sum of a slice for DTLZ problems
#[inline]
pub(super) fn compute_sum<T: Float>(x: &[T]) -> T {
    x.iter().copied().fold(T::zero(), |acc, xi| acc + xi)
}

/// Compute product of a slice for DTLZ problems
///
/// Companion to `compute_sum` above for DTLZ variants whose g-function or
/// shape function needs a product reduction; no currently-implemented
/// problem in this module needs it yet, so it is kept for that future use
/// rather than deleted.
#[inline]
#[allow(dead_code)]
pub(super) fn compute_product<T: Float>(x: &[T]) -> T {
    x.iter().copied().fold(T::one(), |acc, xi| acc * xi)
}

/// Generate uniform points on (M-1)-dimensional simplex
///
/// Used for generating Pareto-optimal points for problems with linear
/// or concave Pareto fronts.
pub(super) fn generate_simplex_points<T: Float>(m: usize, n_points: usize) -> Vec<Vec<T>> {
    if m == 2 {
        // For 2 objectives, simplex is a line segment
        return (0..n_points)
            .map(|i| {
                let t = T::from(i).expect("usize to Float")
                    / T::from(n_points - 1).expect("usize to Float");
                vec![T::one() - t, t]
            })
            .collect();
    }

    // For M > 2, use Das-Dennis approach (simplified uniform sampling)
    let mut points = Vec::new();
    generate_simplex_recursive(m, n_points, T::zero(), Vec::new(), &mut points);
    points
}

/// Recursive helper for simplex point generation
pub(super) fn generate_simplex_recursive<T: Float>(
    m: usize,
    n_points: usize,
    sum: T,
    current: Vec<T>,
    points: &mut Vec<Vec<T>>,
) {
    if current.len() == m - 1 {
        let mut point = current;
        point.push(T::one() - sum);
        points.push(point);
        return;
    }

    let divisions = (n_points as f64).powf(1.0 / (m - 1) as f64).ceil() as usize;
    for i in 0..=divisions {
        let val = T::from(i).expect("usize to Float") / T::from(divisions).expect("usize to Float")
            * (T::one() - sum);
        if sum + val <= T::one() + T::epsilon() {
            let mut next = current.clone();
            next.push(val);
            generate_simplex_recursive(m, n_points, sum + val, next, points);
        }
    }
}
