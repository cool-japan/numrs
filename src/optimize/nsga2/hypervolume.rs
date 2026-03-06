//! Hypervolume indicator calculation for multi-objective optimization.
//!
//! This module provides functions for calculating the hypervolume indicator,
//! which measures the volume of objective space dominated by a Pareto front.

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::cmp::Ordering;

/// Calculate hypervolume indicator for a Pareto front
///
/// # Arguments
///
/// * `front` - Pareto front points (objectives)
/// * `reference_point` - Reference point (must weakly dominate all front points)
///
/// # Returns
///
/// Hypervolume value as `Result<T>`
///
/// # Errors
///
/// Returns error if:
/// - Reference point dimension doesn't match objective space dimension
/// - Reference point doesn't dominate all points
/// - Front is empty
pub fn calculate_hypervolume<T: Float>(front: &[Vec<T>], reference_point: &[T]) -> Result<T> {
    if front.is_empty() {
        return Err(NumRs2Error::ValueError(
            "Pareto front cannot be empty".to_string(),
        ));
    }

    let n_obj = front[0].len();

    if reference_point.len() != n_obj {
        return Err(NumRs2Error::ValueError(format!(
            "Reference point dimension ({}) doesn't match objective space dimension ({})",
            reference_point.len(),
            n_obj
        )));
    }

    // Validate that reference point dominates all front points
    for point in front {
        if point.len() != n_obj {
            return Err(NumRs2Error::ValueError(
                "All points must have the same dimension".to_string(),
            ));
        }

        for (obj_val, ref_val) in point.iter().zip(reference_point.iter()) {
            if obj_val >= ref_val {
                return Err(NumRs2Error::ValueError(
                    "Reference point must weakly dominate all Pareto front points".to_string(),
                ));
            }
        }
    }

    // Dispatch to appropriate algorithm based on dimensionality
    match n_obj {
        2 => hypervolume_2d(front, reference_point),
        3 => hypervolume_3d(front, reference_point),
        _ => hypervolume_wfg(front, reference_point),
    }
}

/// Calculate 2D hypervolume using sweep-line algorithm
fn hypervolume_2d<T: Float>(front: &[Vec<T>], reference_point: &[T]) -> Result<T> {
    let mut points: Vec<(T, T)> = front.iter().map(|p| (p[0], p[1])).collect();

    // Sort by first objective
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut total_volume = T::zero();
    let mut prev_x = reference_point[0];

    for &(x, y) in points.iter().rev() {
        let width = prev_x - x;
        let height = reference_point[1] - y;
        total_volume = total_volume + width * height;
        prev_x = x;
    }

    Ok(total_volume)
}

/// Calculate 3D hypervolume using layer-based algorithm
fn hypervolume_3d<T: Float>(front: &[Vec<T>], reference_point: &[T]) -> Result<T> {
    let mut points: Vec<Vec<T>> = front.to_vec();

    // Sort by first objective (descending)
    points.sort_by(|a, b| b[0].partial_cmp(&a[0]).unwrap_or(Ordering::Equal));

    let mut total_volume = T::zero();
    let mut covered_area = Vec::new();

    for point in &points {
        let x_diff = reference_point[0] - point[0];

        // Calculate uncovered area in the yz-plane
        let area = calculate_uncovered_area_2d(
            &covered_area,
            &[point[1], point[2]],
            &[reference_point[1], reference_point[2]],
        )?;

        total_volume = total_volume + x_diff * area;

        // Update covered area
        covered_area.push(vec![point[1], point[2]]);
    }

    Ok(total_volume)
}

/// Calculate uncovered area in 2D for the 3D hypervolume algorithm
fn calculate_uncovered_area_2d<T: Float>(
    covered: &[Vec<T>],
    point: &[T],
    reference: &[T],
) -> Result<T> {
    if covered.is_empty() {
        return Ok((reference[0] - point[0]) * (reference[1] - point[1]));
    }

    // For simplicity, we use inclusion-exclusion principle
    // This is a simplified version; full WFG would be more efficient
    let mut area = (reference[0] - point[0]) * (reference[1] - point[1]);

    for covered_point in covered {
        // Check if this covered point dominates the current point in 2D
        if covered_point[0] <= point[0] && covered_point[1] <= point[1] {
            // Calculate overlap
            let overlap_width = reference[0] - covered_point[0].max(point[0]);
            let overlap_height = reference[1] - covered_point[1].max(point[1]);
            area = area - overlap_width.max(T::zero()) * overlap_height.max(T::zero());
        }
    }

    Ok(area.max(T::zero()))
}

/// Calculate hypervolume using WFG algorithm for n-dimensional case
///
/// This is a simplified implementation of the WFG algorithm.
/// For production use, consider using a full WFG implementation.
fn hypervolume_wfg<T: Float>(front: &[Vec<T>], reference_point: &[T]) -> Result<T> {
    let n_obj = reference_point.len();

    // Base case: empty front
    if front.is_empty() {
        return Ok(T::zero());
    }

    // Base case: single point
    if front.len() == 1 {
        let mut volume = T::one();
        for (obj_val, ref_val) in front[0].iter().zip(reference_point.iter()) {
            volume = volume * (*ref_val - *obj_val);
        }
        return Ok(volume);
    }

    // Recursive WFG algorithm using inclusion-exclusion principle
    // Find the point with maximum value in the last objective
    let max_idx = front
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a[n_obj - 1]
                .partial_cmp(&b[n_obj - 1])
                .unwrap_or(Ordering::Equal)
        })
        .ok_or_else(|| NumRs2Error::ComputationError("Failed to find maximum point".to_string()))?
        .0;

    let max_point = &front[max_idx];

    // Calculate the exclusive hypervolume contribution of this point
    let mut slice_volume = T::one();
    for (obj_val, ref_val) in max_point.iter().zip(reference_point.iter()) {
        slice_volume = slice_volume * (*ref_val - *obj_val);
    }

    // Create a new reference point for the remaining points
    let mut new_reference = reference_point.to_vec();
    new_reference[n_obj - 1] = max_point[n_obj - 1];

    // Filter points that are not dominated by the new reference point
    let remaining: Vec<Vec<T>> = front
        .iter()
        .enumerate()
        .filter(|(i, point)| {
            *i != max_idx && point.iter().zip(new_reference.iter()).all(|(p, r)| p < r)
        })
        .map(|(_, point)| point.clone())
        .collect();

    // Recursive call
    let remaining_volume = if remaining.is_empty() {
        T::zero()
    } else {
        hypervolume_wfg(&remaining, &new_reference)?
    };

    Ok(slice_volume + remaining_volume)
}
