//! Reference point generation methods for NSGA-III
//!
//! This module provides multiple strategies for generating reference points:
//! - Das-Dennis method: Systematic generation for low-dimensional objective spaces (1-5 objectives)
//! - Two-layer approach: Reduced point count for medium-dimensional spaces (6-8 objectives)
//! - Random sampling: Linear-scaling for high-dimensional spaces (9+ objectives)
//! - Adaptive selection: Automatically chooses the best method based on objective count

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use scirs2_core::random::{thread_rng, Distribution, Uniform};

use super::{vector_norm, ReferencePoint};

/// Generate uniformly distributed reference points using Das-Dennis method
///
/// This creates a structured set of reference points on the unit simplex/hyperplane.
/// For M objectives and p divisions, generates C(M+p-1, p) reference points.
///
/// The Das-Dennis method systematically generates points where each coordinate
/// is a multiple of 1/n_divisions and the sum of all coordinates equals 1.
/// These points are then normalized to lie on the unit hypersphere.
///
/// # Arguments
///
/// * `n_obj` - Number of objectives (M)
/// * `n_divisions` - Number of divisions along each objective axis (p)
///
/// # Returns
///
/// Vector of reference points uniformly distributed on hyperplane
///
/// # Example
///
/// For M=3, p=4: Generates C(6,4)=15 reference points like:
/// - [1.0, 0.0, 0.0]
/// - [0.75, 0.25, 0.0]
/// - [0.75, 0.0, 0.25]
/// - ... etc.
pub(super) fn generate_reference_points<T: Float + std::iter::Sum>(
    n_obj: usize,
    n_divisions: usize,
) -> Result<Vec<ReferencePoint<T>>> {
    if n_obj == 0 {
        return Err(NumRs2Error::ValueError(
            "Number of objectives must be positive".to_string(),
        ));
    }

    if n_divisions == 0 {
        return Err(NumRs2Error::ValueError(
            "Number of divisions must be positive".to_string(),
        ));
    }

    // Generate raw reference points using recursive Das-Dennis method
    let mut raw_points = Vec::new();
    let mut current_point = vec![T::zero(); n_obj];
    generate_recursive_points(
        &mut raw_points,
        &mut current_point,
        n_obj,
        n_divisions,
        n_divisions,
        0,
    )?;

    // Normalize each reference point to unit hypersphere
    let mut reference_points = Vec::with_capacity(raw_points.len());
    for point in raw_points {
        let norm = vector_norm(&point)?;

        let normalized = if norm > T::epsilon() {
            point.iter().map(|&x| x / norm).collect()
        } else {
            // If norm is zero (shouldn't happen), use original point
            point
        };

        reference_points.push(ReferencePoint {
            position: normalized,
            niche_count: 0,
        });
    }

    Ok(reference_points)
}

/// Recursively generate reference points on unit simplex
///
/// This helper function generates all combinations of coordinates that sum to n_divisions,
/// where each coordinate is a non-negative integer between 0 and n_divisions.
///
/// # Arguments
///
/// * `points` - Accumulated reference points
/// * `current` - Current point being constructed
/// * `n_obj` - Total number of objectives
/// * `n_divisions` - Total number of divisions
/// * `remaining` - Remaining sum to distribute
/// * `depth` - Current objective dimension being filled
fn generate_recursive_points<T: Float>(
    points: &mut Vec<Vec<T>>,
    current: &mut Vec<T>,
    n_obj: usize,
    n_divisions: usize,
    remaining: usize,
    depth: usize,
) -> Result<()> {
    // Base case: last objective dimension
    if depth == n_obj - 1 {
        // Assign all remaining to the last dimension
        let value = T::from(remaining).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert division count".to_string())
        })?;
        let divisor = T::from(n_divisions).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert n_divisions".to_string())
        })?;
        current[depth] = value / divisor;

        // Add completed point to list
        points.push(current.clone());
        return Ok(());
    }

    // Recursive case: try all possible values for current dimension
    for i in 0..=remaining {
        let value = T::from(i).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert division value".to_string())
        })?;
        let divisor = T::from(n_divisions).ok_or_else(|| {
            NumRs2Error::ConversionError("Failed to convert n_divisions".to_string())
        })?;
        current[depth] = value / divisor;

        // Recurse with remaining sum
        generate_recursive_points(
            points,
            current,
            n_obj,
            n_divisions,
            remaining - i,
            depth + 1,
        )?;
    }

    Ok(())
}

/// Generate reference points using two-layer approach for 6-8 objectives
///
/// This method reduces the exponential growth of reference points by using:
/// - Outer layer: Boundary points with fewer divisions (captures extremes)
/// - Inner layer: Interior points with scaled divisions (maintains diversity)
///
/// This approach significantly reduces point count while maintaining coverage.
///
/// # Arguments
///
/// * `n_obj` - Number of objectives (recommended for 6-8 objectives)
/// * `n_divisions_outer` - Divisions for outer layer (boundary points)
/// * `n_divisions_inner` - Divisions for inner layer (interior points)
///
/// # Returns
///
/// Vector of reference points on hyperplane
pub(super) fn generate_reference_points_layered<T: Float + std::iter::Sum>(
    n_obj: usize,
    n_divisions_outer: usize,
    n_divisions_inner: usize,
) -> Result<Vec<ReferencePoint<T>>> {
    if n_obj == 0 {
        return Err(NumRs2Error::ValueError(
            "Number of objectives must be positive".to_string(),
        ));
    }

    let mut all_points = Vec::new();

    // Generate outer layer (boundary points with more divisions)
    let mut outer_raw = Vec::new();
    let mut current_point = vec![T::zero(); n_obj];
    generate_recursive_points(
        &mut outer_raw,
        &mut current_point,
        n_obj,
        n_divisions_outer,
        n_divisions_outer,
        0,
    )?;
    all_points.extend(outer_raw);

    // Generate inner layer if divisions specified
    if n_divisions_inner > 0 {
        let mut inner_raw = Vec::new();
        let mut current_point = vec![T::zero(); n_obj];
        generate_recursive_points(
            &mut inner_raw,
            &mut current_point,
            n_obj,
            n_divisions_inner,
            n_divisions_inner,
            0,
        )?;

        // Scale inner points to be interior (multiply by 0.5 to move toward center)
        let half = T::from(0.5)
            .ok_or_else(|| NumRs2Error::ConversionError("Failed to convert 0.5".to_string()))?;

        for point in inner_raw {
            let scaled: Vec<T> = point.iter().map(|&x| x * half).collect();
            all_points.push(scaled);
        }
    }

    // Normalize all points to unit hypersphere
    let mut reference_points = Vec::with_capacity(all_points.len());
    for point in all_points {
        let norm = vector_norm(&point)?;

        let normalized = if norm > T::epsilon() {
            point.iter().map(|&x| x / norm).collect()
        } else {
            point
        };

        reference_points.push(ReferencePoint {
            position: normalized,
            niche_count: 0,
        });
    }

    Ok(reference_points)
}

/// Generate reference points using random sampling for 9+ objectives
///
/// Uses Dirichlet-like sampling to generate uniformly distributed points
/// on the unit simplex. This approach scales linearly with the number of
/// points, avoiding exponential growth.
///
/// # Arguments
///
/// * `n_obj` - Number of objectives
/// * `n_points` - Number of reference points to generate
///
/// # Returns
///
/// Vector of randomly sampled reference points
pub(super) fn generate_reference_points_random<T: Float + std::iter::Sum>(
    n_obj: usize,
    n_points: usize,
) -> Result<Vec<ReferencePoint<T>>> {
    if n_obj == 0 {
        return Err(NumRs2Error::ValueError(
            "Number of objectives must be positive".to_string(),
        ));
    }

    if n_points == 0 {
        return Err(NumRs2Error::ValueError(
            "Number of points must be positive".to_string(),
        ));
    }

    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0, 1.0).map_err(|e| {
        NumRs2Error::ComputationError(format!("Uniform distribution creation failed: {}", e))
    })?;

    let mut reference_points = Vec::with_capacity(n_points);

    for _ in 0..n_points {
        // Generate point on simplex using exponential distribution (Dirichlet with alpha=1)
        let mut point = Vec::with_capacity(n_obj);

        // Sample exponential random variables (using -ln(U) where U ~ Uniform(0,1))
        for _ in 0..n_obj {
            let u: f64 = uniform.sample(&mut rng);
            // Avoid log(0) by clamping
            let u_clamped = u.max(1e-10);
            let exp_sample = -u_clamped.ln();

            let value = T::from(exp_sample).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert exponential sample".to_string())
            })?;
            point.push(value);
        }

        // Normalize to sum to 1 (on simplex)
        let sum: T = point.iter().copied().sum();
        if sum > T::epsilon() {
            for x in &mut point {
                *x = *x / sum;
            }
        }

        // Normalize to unit hypersphere
        let norm = vector_norm(&point)?;
        let normalized = if norm > T::epsilon() {
            point.iter().map(|&x| x / norm).collect()
        } else {
            point
        };

        reference_points.push(ReferencePoint {
            position: normalized,
            niche_count: 0,
        });
    }

    Ok(reference_points)
}

/// Generate reference points using adaptive strategy based on objective count
///
/// Selects the optimal reference point generation method based on the number
/// of objectives to balance quality and computational efficiency:
///
/// - **1-5 objectives**: Full Das-Dennis (systematic, high quality)
/// - **6-8 objectives**: Two-layer approach (reduced points, good coverage)
/// - **9+ objectives**: Random sampling (linear scaling, uniform coverage)
///
/// # Arguments
///
/// * `n_obj` - Number of objectives
/// * `n_divisions` - Number of divisions (interpretation varies by method)
///
/// # Returns
///
/// Vector of reference points generated using the appropriate method
pub(super) fn generate_reference_points_adaptive<T: Float + std::iter::Sum>(
    n_obj: usize,
    n_divisions: usize,
) -> Result<Vec<ReferencePoint<T>>> {
    match n_obj {
        0 => Err(NumRs2Error::ValueError(
            "Number of objectives must be positive".to_string(),
        )),
        1..=5 => {
            // Use full Das-Dennis for 1-5 objectives (proven quality)
            generate_reference_points(n_obj, n_divisions)
        }
        6..=8 => {
            // Use two-layer approach for 6-8 objectives
            // Outer layer: smaller divisions for boundaries
            // Inner layer: even smaller for interior
            let outer_divisions = n_divisions.min(4); // Cap outer divisions
            let inner_divisions = (n_divisions / 2).max(1); // Reduce inner divisions

            generate_reference_points_layered(n_obj, outer_divisions, inner_divisions)
        }
        _ => {
            // For 9+ objectives, use random sampling
            // Generate approximately n_divisions^2 points to maintain some density
            let n_points = (n_divisions * n_divisions).clamp(100, 1000);

            generate_reference_points_random(n_obj, n_points)
        }
    }
}
