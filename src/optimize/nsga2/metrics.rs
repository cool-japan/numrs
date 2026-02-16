//! Quality metrics for multi-objective optimization.
//!
//! This module provides convergence and diversity metrics for evaluating
//! Pareto fronts in multi-objective optimization:
//!
//! ## Convergence Metrics
//! - **IGD (Inverted Generational Distance)**: Coverage of reference front
//! - **GD (Generational Distance)**: Convergence to reference front
//!
//! ## Diversity Metrics
//! - **Spacing (S)**: Uniformity of distribution
//! - **Spread (Delta)**: Extent and uniformity of spread

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::cmp::Ordering;

// =============================================================================
// Utility Functions
// =============================================================================

/// Calculate Euclidean distance between two points
///
/// # Arguments
///
/// * `a` - First point
/// * `b` - Second point
///
/// # Returns
///
/// Euclidean distance between a and b
pub(crate) fn euclidean_distance<T: Float + std::iter::Sum>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (*ai - *bi) * (*ai - *bi))
        .sum::<T>()
        .sqrt()
}

/// Calculate minimum distance from a point to a Pareto front
///
/// # Arguments
///
/// * `point` - Point to measure distance from
/// * `front` - Pareto front to measure distance to
///
/// # Returns
///
/// Minimum Euclidean distance to any point in the front
fn min_distance_to_front<T: Float + std::iter::Sum>(point: &[T], front: &[Vec<T>]) -> T {
    front
        .iter()
        .map(|front_point| euclidean_distance(point, front_point))
        .fold(T::infinity(), |min_dist, dist| {
            if dist < min_dist {
                dist
            } else {
                min_dist
            }
        })
}

// =============================================================================
// Convergence Metrics
// =============================================================================

/// Calculate Inverted Generational Distance (IGD)
///
/// IGD measures how well the obtained front covers the reference front.
/// It calculates the average distance from reference points to the obtained front.
/// Lower values indicate better convergence and coverage.
///
/// # Formula
///
/// IGD = (1/|P_ref|) * sqrt(sum(d_i^2))
///
/// where d_i is the minimum distance from reference point i to the obtained front
///
/// # Arguments
///
/// * `obtained_front` - Obtained Pareto front objective values
/// * `reference_front` - True/reference Pareto front objective values
///
/// # Returns
///
/// IGD metric value
///
/// # Errors
///
/// Returns error if:
/// - Either front is empty
/// - Points have inconsistent dimensions
///
/// # Interpretation
///
/// - IGD = 0: Perfect convergence to reference front
/// - Lower values: Better convergence and coverage
/// - Higher values: Poor convergence or incomplete coverage
/// - Typical range: [0, infinity)
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::calculate_igd;
///
/// let obtained = vec![
///     vec![1.0, 3.0],
///     vec![2.0, 2.0],
///     vec![3.0, 1.0],
/// ];
///
/// let reference = vec![
///     vec![1.0, 3.0],
///     vec![2.0, 2.0],
///     vec![3.0, 1.0],
/// ];
///
/// let igd = calculate_igd(&obtained, &reference).expect("IGD calculation should succeed");
/// assert!(igd >= 0.0);
/// ```
pub fn calculate_igd<T: Float + std::fmt::Display + std::iter::Sum>(
    obtained_front: &[Vec<T>],
    reference_front: &[Vec<T>],
) -> Result<T> {
    if obtained_front.is_empty() {
        return Err(NumRs2Error::ValueError(
            "Obtained front cannot be empty".to_string(),
        ));
    }

    if reference_front.is_empty() {
        return Err(NumRs2Error::ValueError(
            "Reference front cannot be empty".to_string(),
        ));
    }

    let n_obj = obtained_front[0].len();

    // Validate dimensions
    for point in obtained_front {
        if point.len() != n_obj {
            return Err(NumRs2Error::ValueError(
                "All obtained points must have the same dimension".to_string(),
            ));
        }
    }

    for point in reference_front {
        if point.len() != n_obj {
            return Err(NumRs2Error::ValueError(
                "All reference points must have the same dimension".to_string(),
            ));
        }
    }

    // Calculate sum of squared minimum distances from reference to obtained
    let sum_squared_distances: T = reference_front
        .iter()
        .map(|ref_point| {
            let min_dist = min_distance_to_front(ref_point, obtained_front);
            min_dist * min_dist
        })
        .sum();

    // Calculate IGD
    let n_ref = T::from(reference_front.len()).ok_or_else(|| {
        NumRs2Error::ConversionError("Failed to convert reference front size to Float".to_string())
    })?;

    Ok((sum_squared_distances / n_ref).sqrt())
}

/// Calculate Generational Distance (GD)
///
/// GD measures the convergence of the obtained front to the reference front.
/// It calculates the average distance from obtained points to the reference front.
/// Lower values indicate better convergence.
///
/// # Formula
///
/// GD = (1/n)^(1/p) * (sum(d_i^p))^(1/p)
///
/// where:
/// - d_i is the minimum distance from obtained point i to the reference front
/// - p is typically 2 (default)
///
/// # Arguments
///
/// * `obtained_front` - Obtained Pareto front objective values
/// * `reference_front` - True/reference Pareto front objective values
/// * `p` - Power parameter (typically 2); if None, defaults to 2
///
/// # Returns
///
/// GD metric value
///
/// # Errors
///
/// Returns error if:
/// - Either front is empty
/// - Points have inconsistent dimensions
/// - p <= 0
///
/// # Interpretation
///
/// - GD = 0: Perfect convergence to reference front
/// - Lower values: Better convergence
/// - Higher values: Poor convergence
/// - Typical range: [0, infinity)
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::calculate_gd;
///
/// let obtained = vec![
///     vec![1.0, 3.0],
///     vec![2.0, 2.0],
///     vec![3.0, 1.0],
/// ];
///
/// let reference = vec![
///     vec![1.0, 3.0],
///     vec![2.0, 2.0],
///     vec![3.0, 1.0],
/// ];
///
/// let gd = calculate_gd(&obtained, &reference, None).expect("GD calculation should succeed");
/// assert!(gd >= 0.0);
/// ```
pub fn calculate_gd<T: Float + std::fmt::Display + std::iter::Sum>(
    obtained_front: &[Vec<T>],
    reference_front: &[Vec<T>],
    p: Option<T>,
) -> Result<T> {
    if obtained_front.is_empty() {
        return Err(NumRs2Error::ValueError(
            "Obtained front cannot be empty".to_string(),
        ));
    }

    if reference_front.is_empty() {
        return Err(NumRs2Error::ValueError(
            "Reference front cannot be empty".to_string(),
        ));
    }

    let p_val = p.unwrap_or_else(|| {
        T::from(2.0).expect("Default p=2.0 should convert to Float")
    });

    if p_val <= T::zero() {
        return Err(NumRs2Error::ValueError(
            "Power parameter p must be positive".to_string(),
        ));
    }

    let n_obj = obtained_front[0].len();

    // Validate dimensions
    for point in obtained_front {
        if point.len() != n_obj {
            return Err(NumRs2Error::ValueError(
                "All obtained points must have the same dimension".to_string(),
            ));
        }
    }

    for point in reference_front {
        if point.len() != n_obj {
            return Err(NumRs2Error::ValueError(
                "All reference points must have the same dimension".to_string(),
            ));
        }
    }

    // Calculate sum of powered minimum distances from obtained to reference
    let sum_powered_distances: T = obtained_front
        .iter()
        .map(|obtained_point| {
            let min_dist = min_distance_to_front(obtained_point, reference_front);
            min_dist.powf(p_val)
        })
        .sum();

    // Calculate GD
    let n_obtained = T::from(obtained_front.len()).ok_or_else(|| {
        NumRs2Error::ConversionError("Failed to convert obtained front size to Float".to_string())
    })?;

    Ok((sum_powered_distances / n_obtained).powf(T::one() / p_val))
}

// =============================================================================
// Diversity Metrics
// =============================================================================

/// Calculate spacing metric for Pareto front
///
/// Spacing (S) measures the uniformity of distribution in the Pareto front.
/// Lower values indicate better uniformity.
///
/// # Formula
///
/// S = sqrt(1/(n-1) * sum((d_i - d_mean)^2))
///
/// where d_i is the minimum Euclidean distance from point i to other points
///
/// # Arguments
///
/// * `front` - Pareto front objective values
///
/// # Returns
///
/// Spacing metric value
///
/// # Errors
///
/// Returns error if:
/// - Front has fewer than 2 points
/// - Points have inconsistent dimensions
///
/// # Interpretation
///
/// - S = 0: Perfectly uniform distribution
/// - Lower values: Better uniformity
/// - Higher values: Clustered or uneven distribution
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::calculate_spacing;
///
/// let front = vec![
///     vec![1.0, 3.0],
///     vec![2.0, 2.0],
///     vec![3.0, 1.0],
/// ];
///
/// let spacing = calculate_spacing(&front).expect("Spacing calculation should succeed");
/// assert!(spacing >= 0.0);
/// ```
pub fn calculate_spacing<T: Float + std::fmt::Display + std::iter::Sum>(front: &[Vec<T>]) -> Result<T> {
    if front.len() < 2 {
        return Err(NumRs2Error::ValueError(
            "Spacing requires at least 2 points".to_string(),
        ));
    }

    let n = front.len();
    let n_obj = front[0].len();

    // Validate dimensions
    for point in front {
        if point.len() != n_obj {
            return Err(NumRs2Error::ValueError(
                "All points must have the same dimension".to_string(),
            ));
        }
    }

    // Calculate minimum distances
    let mut min_distances = Vec::with_capacity(n);

    for i in 0..n {
        let mut min_dist = T::infinity();

        for j in 0..n {
            if i != j {
                let dist = euclidean_distance(&front[i], &front[j]);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }

        min_distances.push(min_dist);
    }

    // Calculate mean distance
    let mean_dist = min_distances.iter().fold(T::zero(), |acc, &d| acc + d)
        / T::from(n).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert n to Float".to_string())
        })?;

    // Calculate variance
    let variance = min_distances
        .iter()
        .map(|&d| (d - mean_dist) * (d - mean_dist))
        .sum::<T>()
        / T::from(n - 1).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert n-1 to Float".to_string())
        })?;

    Ok(variance.sqrt())
}

/// Find extreme points in each objective dimension
///
/// # Arguments
///
/// * `front` - Pareto front objective values
///
/// # Returns
///
/// Vector of extreme points (one for each objective)
fn find_extreme_points<T: Float + Clone>(front: &[Vec<T>]) -> Vec<Vec<T>> {
    if front.is_empty() {
        return Vec::new();
    }

    let n_obj = front[0].len();
    let mut extremes = Vec::with_capacity(n_obj);

    for obj_idx in 0..n_obj {
        // Find point with minimum value in this objective
        let mut min_point = &front[0];
        let mut min_val = front[0][obj_idx];

        for point in front {
            if point[obj_idx] < min_val {
                min_val = point[obj_idx];
                min_point = point;
            }
        }

        extremes.push(min_point.clone());
    }

    extremes
}

/// Calculate spread metric for Pareto front
///
/// Spread (Delta) measures both the extent of spread and distribution uniformity.
/// Lower values indicate better spread and uniformity.
///
/// # Formula
///
/// Delta = (d_f + d_l + sum|d_i - d_mean|) / (d_f + d_l + (n-1)*d_mean)
///
/// where:
/// - d_f, d_l = distances to extreme points
/// - d_i = consecutive distances after sorting
/// - d_mean = mean of consecutive distances
///
/// # Arguments
///
/// * `front` - Pareto front objective values
/// * `extreme_points` - Optional known extreme points; if None, computed automatically
///
/// # Returns
///
/// Spread metric value
///
/// # Errors
///
/// Returns error if:
/// - Front has fewer than 2 points
/// - Points have inconsistent dimensions
///
/// # Interpretation
///
/// - Delta = 0: Perfect spread and uniformity
/// - Lower values: Better spread
/// - Higher values: Poor extent or uneven distribution
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::calculate_spread;
///
/// let front = vec![
///     vec![1.0, 3.0],
///     vec![2.0, 2.0],
///     vec![3.0, 1.0],
/// ];
///
/// let spread = calculate_spread(&front, None).expect("Spread calculation should succeed");
/// assert!(spread >= 0.0);
/// ```
pub fn calculate_spread<T: Float + std::fmt::Display + std::iter::Sum>(
    front: &[Vec<T>],
    extreme_points: Option<&[Vec<T>]>,
) -> Result<T> {
    if front.len() < 2 {
        return Err(NumRs2Error::ValueError(
            "Spread requires at least 2 points".to_string(),
        ));
    }

    let n = front.len();
    let n_obj = front[0].len();

    // Validate dimensions
    for point in front {
        if point.len() != n_obj {
            return Err(NumRs2Error::ValueError(
                "All points must have the same dimension".to_string(),
            ));
        }
    }

    // Get or compute extreme points
    let extremes = if let Some(ext) = extreme_points {
        ext.to_vec()
    } else {
        find_extreme_points(front)
    };

    // Sort front by first objective for consecutive distance calculation
    let mut sorted_front = front.to_vec();
    sorted_front.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(Ordering::Equal));

    // Calculate consecutive distances
    let mut consecutive_distances = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let dist = euclidean_distance(&sorted_front[i], &sorted_front[i + 1]);
        consecutive_distances.push(dist);
    }

    // Calculate mean consecutive distance
    let d_mean = consecutive_distances.iter().copied().sum::<T>()
        / T::from(consecutive_distances.len()).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert length to Float".to_string())
        })?;

    // Calculate distances to extreme points
    let d_f = euclidean_distance(&sorted_front[0], &extremes[0]);
    let d_l = euclidean_distance(&sorted_front[n - 1], &extremes[n_obj - 1]);

    // Calculate sum of absolute deviations
    let sum_deviations: T = consecutive_distances
        .iter()
        .map(|&d| (d - d_mean).abs())
        .sum();

    // Calculate spread
    let numerator = d_f + d_l + sum_deviations;
    let denominator = d_f + d_l
        + d_mean
            * T::from(n - 1).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert n-1 to Float".to_string())
            })?;

    if denominator == T::zero() {
        return Ok(T::zero());
    }

    Ok(numerator / denominator)
}
