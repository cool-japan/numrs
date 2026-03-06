//! Pareto front extraction, validation, and utility functions.
//!
//! This module provides functions for working with Pareto fronts:
//! - Pareto optimality checking
//! - Front validation
//! - Non-dominated solution extraction
//! - Front sorting and filtering

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::cmp::Ordering;

use super::{dominates, Individual};

/// Check if a solution is Pareto optimal relative to a front
///
/// # Arguments
///
/// * `solution` - Objective values to check
/// * `front` - Pareto front to compare against
///
/// # Returns
///
/// `true` if solution is not dominated by any front member, `false` otherwise
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::is_pareto_optimal;
///
/// let solution = vec![1.0, 2.0];
/// let front = vec![
///     vec![2.0, 1.0],
///     vec![1.5, 1.5],
/// ];
///
/// assert!(is_pareto_optimal(&solution, &front));
/// ```
pub fn is_pareto_optimal<T: Float>(solution: &[T], front: &[Vec<T>]) -> bool {
    // Check if any point in the front dominates this solution
    for point in front {
        if dominates(point, solution) {
            return false;
        }
    }
    true
}

/// Validate that a set of solutions forms a valid Pareto front
///
/// A valid Pareto front must satisfy:
/// 1. Contains at least one solution
/// 2. No solution dominates another in the front
///
/// # Arguments
///
/// * `front` - Solutions to validate
///
/// # Returns
///
/// `Ok(true)` if valid Pareto front, error otherwise
///
/// # Errors
///
/// Returns error if:
/// - Front is empty
/// - Any solution dominates another in the front
/// - Solutions have inconsistent dimensions
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::validate_pareto_front;
///
/// let front = vec![
///     vec![1.0, 2.0],
///     vec![2.0, 1.0],
/// ];
///
/// assert!(validate_pareto_front(&front).is_ok());
/// ```
pub fn validate_pareto_front<T: Float>(front: &[Vec<T>]) -> Result<bool> {
    if front.is_empty() {
        return Err(NumRs2Error::ValueError(
            "Pareto front cannot be empty".to_string(),
        ));
    }

    // Check dimension consistency
    let n_obj = front[0].len();
    for point in front {
        if point.len() != n_obj {
            return Err(NumRs2Error::ValueError(
                "All solutions must have the same number of objectives".to_string(),
            ));
        }
    }

    // Check that no solution dominates another
    for (i, solution_i) in front.iter().enumerate() {
        for (j, solution_j) in front.iter().enumerate() {
            if i != j && dominates(solution_i, solution_j) {
                return Err(NumRs2Error::ValueError(format!(
                    "Invalid Pareto front: solution {} dominates solution {}",
                    i, j
                )));
            }
        }
    }

    Ok(true)
}

/// Extract non-dominated solutions from a set
///
/// Filters out all dominated solutions, returning only those
/// that form the Pareto front.
///
/// # Arguments
///
/// * `solutions` - Set of solutions to filter
///
/// # Returns
///
/// Vector of non-dominated solutions
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::extract_non_dominated;
///
/// let solutions = vec![
///     vec![1.0, 3.0],  // Non-dominated
///     vec![2.0, 2.0],  // Non-dominated
///     vec![3.0, 1.0],  // Non-dominated
///     vec![2.5, 2.5],  // Dominated by (2.0, 2.0)
/// ];
///
/// let front = extract_non_dominated(&solutions);
/// assert_eq!(front.len(), 3);
/// ```
pub fn extract_non_dominated<T: Float + Clone>(solutions: &[Vec<T>]) -> Vec<Vec<T>> {
    let mut front: Vec<Vec<T>> = Vec::new();

    for solution in solutions {
        let mut is_dominated = false;

        // Check if this solution is dominated by any existing front member
        for front_member in &front {
            if dominates(front_member, solution) {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            // Remove any front members dominated by this solution
            front.retain(|front_member| !dominates(solution, front_member));

            // Add this solution to the front
            front.push(solution.clone());
        }
    }

    front
}

// =============================================================================
// Enhanced Pareto Front Extraction
// =============================================================================

/// Extract Pareto front from population
///
/// Extracts all rank-0 individuals and sorts them by first objective
/// for consistent ordering.
///
/// # Arguments
///
/// * `population` - Population of individuals
///
/// # Returns
///
/// Vector of Pareto-optimal individuals (rank 0), sorted by first objective
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::{Individual, extract_pareto_front};
///
/// let population = vec![
///     Individual {
///         variables: vec![1.0],
///         objectives: vec![1.0, 3.0],
///         rank: 0,
///         crowding_distance: 0.0,
///     },
///     Individual {
///         variables: vec![2.0],
///         objectives: vec![2.0, 2.0],
///         rank: 1,
///         crowding_distance: 0.0,
///     },
/// ];
///
/// let front = extract_pareto_front(&population);
/// assert_eq!(front.len(), 1);
/// ```
pub fn extract_pareto_front<T: Float>(population: &[Individual<T>]) -> Vec<Individual<T>> {
    let mut front: Vec<Individual<T>> = population
        .iter()
        .filter(|ind| ind.rank == 0)
        .cloned()
        .collect();

    // Sort by first objective for consistent ordering
    front.sort_by(|a, b| {
        a.objectives[0]
            .partial_cmp(&b.objectives[0])
            .unwrap_or(Ordering::Equal)
    });

    front
}

/// Extract objective values from Pareto front
///
/// Extracts only the objective values from rank-0 individuals,
/// useful for metric calculations.
///
/// # Arguments
///
/// * `population` - Population of individuals
///
/// # Returns
///
/// Vector of objective value vectors from Pareto front
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::{Individual, extract_front_objectives};
///
/// let population = vec![
///     Individual {
///         variables: vec![1.0],
///         objectives: vec![1.0, 3.0],
///         rank: 0,
///         crowding_distance: 0.0,
///     },
///     Individual {
///         variables: vec![2.0],
///         objectives: vec![2.0, 2.0],
///         rank: 1,
///         crowding_distance: 0.0,
///     },
/// ];
///
/// let objectives = extract_front_objectives(&population);
/// assert_eq!(objectives.len(), 1);
/// assert_eq!(objectives[0], vec![1.0, 3.0]);
/// ```
pub fn extract_front_objectives<T: Float>(population: &[Individual<T>]) -> Vec<Vec<T>> {
    population
        .iter()
        .filter(|ind| ind.rank == 0)
        .map(|ind| ind.objectives.clone())
        .collect()
}

/// Sort Pareto front by specific objective
///
/// Sorts a Pareto front by a specific objective dimension.
/// Useful for visualization and analysis.
///
/// # Arguments
///
/// * `front` - Pareto front to sort (modified in place)
/// * `objective_idx` - Index of objective to sort by
///
/// # Panics
///
/// Panics if `objective_idx` is out of bounds
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::{Individual, sort_front_by_objective};
///
/// let mut front = vec![
///     Individual {
///         variables: vec![2.0],
///         objectives: vec![2.0, 2.0],
///         rank: 0,
///         crowding_distance: 0.0,
///     },
///     Individual {
///         variables: vec![1.0],
///         objectives: vec![1.0, 3.0],
///         rank: 0,
///         crowding_distance: 0.0,
///     },
/// ];
///
/// sort_front_by_objective(&mut front, 0);
/// assert_eq!(front[0].objectives[0], 1.0);
/// ```
pub fn sort_front_by_objective<T: Float>(front: &mut [Individual<T>], objective_idx: usize) {
    front.sort_by(|a, b| {
        a.objectives[objective_idx]
            .partial_cmp(&b.objectives[objective_idx])
            .unwrap_or(Ordering::Equal)
    });
}

/// Filter dominated solutions from a set of individuals
///
/// Removes all dominated solutions and optionally re-ranks remaining ones.
///
/// # Arguments
///
/// * `individuals` - Set of individuals to filter
///
/// # Returns
///
/// Vector of non-dominated individuals with updated ranks
///
/// # Example
///
/// ```
/// use numrs2::optimize::nsga2::{Individual, filter_dominated_solutions};
///
/// let individuals = vec![
///     Individual {
///         variables: vec![1.0],
///         objectives: vec![1.0, 3.0],
///         rank: 0,
///         crowding_distance: 0.0,
///     },
///     Individual {
///         variables: vec![2.0],
///         objectives: vec![2.0, 2.0],
///         rank: 0,
///         crowding_distance: 0.0,
///     },
///     Individual {
///         variables: vec![3.0],
///         objectives: vec![3.0, 3.0],
///         rank: 1,
///         crowding_distance: 0.0,
///     },
/// ];
///
/// let filtered = filter_dominated_solutions(&individuals);
/// assert!(filtered.len() <= individuals.len());
/// ```
pub fn filter_dominated_solutions<T: Float>(individuals: &[Individual<T>]) -> Vec<Individual<T>> {
    let mut non_dominated: Vec<Individual<T>> = Vec::new();

    for individual in individuals {
        let mut is_dominated = false;

        // Check if this individual is dominated by any non-dominated individual
        for nd_ind in &non_dominated {
            if dominates(&nd_ind.objectives, &individual.objectives) {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            // Remove any non-dominated individuals that this one dominates
            non_dominated.retain(|nd_ind| !dominates(&individual.objectives, &nd_ind.objectives));

            // Add this individual
            let mut new_ind = individual.clone();
            new_ind.rank = 0;
            non_dominated.push(new_ind);
        }
    }

    non_dominated
}
