//! NSGA-III: Non-dominated Sorting Genetic Algorithm III
//!
//! NSGA-III is a reference point-based multi-objective evolutionary algorithm designed
//! for many-objective optimization problems (3+ objectives). Unlike NSGA-II which uses
//! crowding distance, NSGA-III employs systematically distributed reference points to
//! maintain population diversity.
//!
//! # Key Features
//!
//! - **Reference Points**: Uses Das-Dennis method for systematic reference point generation
//! - **Association Mechanism**: Associates each solution with the nearest reference point
//! - **Niche Preservation**: Selects solutions based on niche count around reference points
//! - **Many-Objective Optimization**: Effective for problems with 3 or more objectives
//! - **Adaptive Normalization**: Normalizes objectives for fair distance calculations
//!
//! # Algorithm Overview
//!
//! 1. Generate uniformly distributed reference points on hyperplane
//! 2. Initialize population and evaluate objectives
//! 3. For each generation:
//!    - Create offspring through selection, crossover, mutation
//!    - Combine parent and offspring populations
//!    - Perform non-dominated sorting
//!    - Normalize objectives in each front
//!    - Associate solutions with reference points
//!    - Select solutions using niche preservation
//!
//! # Example
//!
//! ```
//! use numrs2::optimize::nsga3::{nsga3, NSGA3Config};
//!
//! // Minimize three objectives: f1(x) = x^2, f2(x) = (x-1)^2, f3(x) = (x-2)^2
//! let objectives = vec![
//!     |x: &[f64]| x[0] * x[0],
//!     |x: &[f64]| (x[0] - 1.0).powi(2),
//!     |x: &[f64]| (x[0] - 2.0).powi(2),
//! ];
//!
//! let bounds = vec![(0.0, 3.0)];
//! let config = NSGA3Config::default();
//!
//! let result = nsga3(&objectives, &bounds, Some(config))
//!     .expect("NSGA-III should succeed");
//! ```
//!
//! # References
//!
//! K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using
//! Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With
//! Box Constraints," IEEE Transactions on Evolutionary Computation, vol. 18, no. 4,
//! pp. 577-601, Aug. 2014.

mod reference_points;

#[cfg(test)]
mod tests;

use reference_points::*;

use crate::error::{NumRs2Error, Result};
use lazy_static::lazy_static;
use num_traits::Float;
use scirs2_core::random::{thread_rng, Distribution, Rng, Uniform};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Mutex;

/// Type alias for reference point cache
type ReferencePointCache<T> = Mutex<HashMap<(usize, usize, u8), Vec<ReferencePoint<T>>>>;

// Global cache for reference points to avoid regenerating common configurations
// Key: (n_objectives, n_divisions, method_id) where method_id: 0=das-dennis, 1=layered, 2=random
lazy_static! {
    static ref REFERENCE_POINT_CACHE_F64: ReferencePointCache<f64> = Mutex::new(HashMap::new());
    static ref REFERENCE_POINT_CACHE_F32: ReferencePointCache<f32> = Mutex::new(HashMap::new());
}

/// Configuration for NSGA-III
#[derive(Debug, Clone)]
pub struct NSGA3Config<T: Float> {
    /// Population size (should be compatible with reference points)
    pub pop_size: usize,
    /// Number of generations
    pub max_generations: usize,
    /// Crossover probability
    pub crossover_rate: T,
    /// Mutation probability
    pub mutation_rate: T,
    /// Distribution index for crossover (SBX)
    pub eta_c: T,
    /// Distribution index for mutation
    pub eta_m: T,
    /// Number of divisions for Das-Dennis reference point generation
    pub n_divisions: usize,
}

impl<T: Float> Default for NSGA3Config<T> {
    fn default() -> Self {
        Self {
            pop_size: 92, // Suitable for 3 objectives with 12 divisions
            max_generations: 100,
            crossover_rate: T::from(0.9).expect("0.9 should convert to Float"),
            mutation_rate: T::from(0.1).expect("0.1 should convert to Float"),
            eta_c: T::from(30.0).expect("30.0 should convert to Float"),
            eta_m: T::from(20.0).expect("20.0 should convert to Float"),
            n_divisions: 12,
        }
    }
}

/// Individual in the population
#[derive(Clone, Debug)]
pub struct Individual<T: Float> {
    /// Decision variables
    pub variables: Vec<T>,
    /// Objective values
    pub objectives: Vec<T>,
    /// Domination rank (0 = non-dominated front)
    pub rank: usize,
    /// Associated reference point index
    pub reference_point_index: Option<usize>,
    /// Perpendicular distance to associated reference line
    pub perpendicular_distance: T,
}

/// Reference point for guiding population distribution
#[derive(Clone, Debug)]
pub struct ReferencePoint<T: Float> {
    /// Position in objective space (on unit hyperplane)
    pub position: Vec<T>,
    /// Number of solutions associated with this reference point
    pub niche_count: usize,
}

/// Result of NSGA-III optimization
#[derive(Debug)]
pub struct NSGA3Result<T: Float> {
    /// Pareto-optimal solutions (rank 0)
    pub pareto_front: Vec<Individual<T>>,
    /// All final population
    pub population: Vec<Individual<T>>,
    /// Number of generations executed
    pub generations: usize,
    /// Generated reference points
    pub reference_points: Vec<ReferencePoint<T>>,
}

/// NSGA-III multi-objective optimization
///
/// # Arguments
///
/// * `objectives` - Vector of objective functions to minimize
/// * `bounds` - Parameter bounds as (lower, upper) tuples
/// * `config` - Optional NSGA-III configuration
///
/// # Returns
///
/// `NSGA3Result` containing Pareto-optimal solutions and reference points
///
/// # Errors
///
/// Returns error if:
/// - Less than 2 objectives provided
/// - Empty bounds
/// - Invalid population size
/// - Reference point generation fails
pub fn nsga3<T, F>(
    objectives: &[F],
    bounds: &[(T, T)],
    config: Option<NSGA3Config<T>>,
) -> Result<NSGA3Result<T>>
where
    T: Float + std::fmt::Display + std::iter::Sum,
    F: Fn(&[T]) -> T,
{
    let config = config.unwrap_or_default();
    let n_obj = objectives.len();
    let n_var = bounds.len();

    // Validate inputs
    if n_obj < 2 {
        return Err(NumRs2Error::ValueError(
            "NSGA-III requires at least 2 objectives".to_string(),
        ));
    }

    if n_var == 0 {
        return Err(NumRs2Error::ValueError(
            "Bounds must have at least one dimension".to_string(),
        ));
    }

    if config.pop_size < 4 {
        return Err(NumRs2Error::ValueError(
            "Population size must be at least 4".to_string(),
        ));
    }

    // Generate reference points using adaptive method based on objective count
    // - 1-5 objectives: Full Das-Dennis (systematic coverage)
    // - 6-8 objectives: Two-layer approach (reduced exponential growth)
    // - 9+ objectives: Random sampling (linear scaling)
    let reference_points = generate_reference_points_adaptive(n_obj, config.n_divisions)?;

    let mut rng = thread_rng();

    // Initialize population
    let mut population = initialize_population(objectives, bounds, config.pop_size, &mut rng)?;

    // Main evolutionary loop
    for _generation in 0..config.max_generations {
        // Create offspring through selection, crossover, and mutation
        let mut offspring = Vec::with_capacity(config.pop_size);

        while offspring.len() < config.pop_size {
            // Binary tournament selection
            let parent1 = tournament_selection(&population, &mut rng)?;
            let parent2 = tournament_selection(&population, &mut rng)?;

            // Simulated Binary Crossover (SBX)
            let (mut child1, mut child2) = if T::from(rng.gen::<f64>()).ok_or_else(|| {
                NumRs2Error::ConversionError("Random value conversion failed".to_string())
            })? < config.crossover_rate
            {
                sbx_crossover(
                    &parent1.variables,
                    &parent2.variables,
                    bounds,
                    config.eta_c,
                    &mut rng,
                )?
            } else {
                (parent1.variables.clone(), parent2.variables.clone())
            };

            // Polynomial mutation
            if T::from(rng.gen::<f64>()).ok_or_else(|| {
                NumRs2Error::ConversionError("Random value conversion failed".to_string())
            })? < config.mutation_rate
            {
                polynomial_mutation(&mut child1, bounds, config.eta_m, &mut rng)?;
            }

            if T::from(rng.gen::<f64>()).ok_or_else(|| {
                NumRs2Error::ConversionError("Random value conversion failed".to_string())
            })? < config.mutation_rate
            {
                polynomial_mutation(&mut child2, bounds, config.eta_m, &mut rng)?;
            }

            // Evaluate offspring
            offspring.push(create_individual(&child1, objectives));
            if offspring.len() < config.pop_size {
                offspring.push(create_individual(&child2, objectives));
            }
        }

        // Combine parent and offspring populations
        population.extend(offspring);

        // Environmental selection using reference points
        population =
            environmental_selection(population, &reference_points, config.pop_size, n_obj)?;
    }

    // Extract Pareto front (rank 0)
    let pareto_front: Vec<Individual<T>> = population
        .iter()
        .filter(|ind| ind.rank == 0)
        .cloned()
        .collect();

    Ok(NSGA3Result {
        pareto_front,
        population,
        generations: config.max_generations,
        reference_points,
    })
}

/// Calculate L2 norm of a vector
fn vector_norm<T: Float + std::iter::Sum>(v: &[T]) -> Result<T> {
    if v.is_empty() {
        return Err(NumRs2Error::ValueError(
            "Cannot compute norm of empty vector".to_string(),
        ));
    }

    let sum_squares: T = v.iter().map(|&x| x * x).sum();
    Ok(sum_squares.sqrt())
}

/// Initialize random population
fn initialize_population<T, F>(
    objectives: &[F],
    bounds: &[(T, T)],
    pop_size: usize,
    rng: &mut impl Rng,
) -> Result<Vec<Individual<T>>>
where
    T: Float + std::fmt::Display,
    F: Fn(&[T]) -> T,
{
    let n_var = bounds.len();
    let mut population = Vec::with_capacity(pop_size);

    for _ in 0..pop_size {
        let mut variables = Vec::with_capacity(n_var);

        for &(lower, upper) in bounds {
            let lower_f64 = lower.to_f64().ok_or_else(|| {
                NumRs2Error::ConversionError("Bound conversion failed".to_string())
            })?;
            let upper_f64 = upper.to_f64().ok_or_else(|| {
                NumRs2Error::ConversionError("Bound conversion failed".to_string())
            })?;

            let uniform = Uniform::new(lower_f64, upper_f64).map_err(|e| {
                NumRs2Error::ComputationError(format!("Uniform creation failed: {}", e))
            })?;

            let value = T::from(uniform.sample(rng)).ok_or_else(|| {
                NumRs2Error::ConversionError("Sample conversion failed".to_string())
            })?;

            variables.push(value);
        }

        population.push(create_individual(&variables, objectives));
    }

    Ok(population)
}

/// Create individual with evaluated objectives
fn create_individual<T, F>(variables: &[T], objectives: &[F]) -> Individual<T>
where
    T: Float,
    F: Fn(&[T]) -> T,
{
    let obj_values: Vec<T> = objectives.iter().map(|f| f(variables)).collect();

    Individual {
        variables: variables.to_vec(),
        objectives: obj_values,
        rank: 0,
        reference_point_index: None,
        perpendicular_distance: T::infinity(),
    }
}

/// Environmental selection using reference points
///
/// Implements the NSGA-III environmental selection:
/// 1. Non-dominated sorting
/// 2. Normalize objectives
/// 3. Associate solutions with reference points
/// 4. Niche preservation selection
fn environmental_selection<T: Float + std::iter::Sum>(
    mut population: Vec<Individual<T>>,
    reference_points: &[ReferencePoint<T>],
    target_size: usize,
    n_obj: usize,
) -> Result<Vec<Individual<T>>> {
    // Make a mutable copy of reference points to track niche counts
    let mut ref_points_copy: Vec<ReferencePoint<T>> = reference_points.to_vec();

    // Perform non-dominated sorting
    fast_non_dominated_sort(&mut population);

    // If first front is smaller than target, include entire fronts
    let mut selected = Vec::new();
    let mut front_idx = 0;

    loop {
        let current_front: Vec<Individual<T>> = population
            .iter()
            .filter(|ind| ind.rank == front_idx)
            .cloned()
            .collect();

        if current_front.is_empty() {
            break;
        }

        if selected.len() + current_front.len() <= target_size {
            // Include entire front
            let mut front_to_add = current_front;

            // Normalize and associate even for fully included fronts
            normalize_objectives(&mut front_to_add, n_obj)?;
            associate_with_reference_points(&mut front_to_add, &ref_points_copy)?;

            // Update niche counts for selected individuals
            update_niche_counts(&front_to_add, &mut ref_points_copy);

            selected.extend(front_to_add);
            front_idx += 1;
        } else {
            // Last front to consider - need to select K individuals
            let k = target_size - selected.len();

            // Normalize objectives and associate with reference points
            let mut last_front = current_front;
            normalize_objectives(&mut last_front, n_obj)?;
            associate_with_reference_points(&mut last_front, &ref_points_copy)?;

            // Select K solutions using niche preservation
            let selected_from_last = niche_preservation_selection(last_front, &ref_points_copy, k)?;
            selected.extend(selected_from_last);
            break;
        }
    }

    Ok(selected)
}

/// Fast non-dominated sorting (same as NSGA-II)
fn fast_non_dominated_sort<T: Float>(population: &mut [Individual<T>]) {
    let n = population.len();
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut domination_count = vec![0; n];
    let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];

    // First front
    let mut current_front = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }

            if dominates(&population[i].objectives, &population[j].objectives) {
                dominated_solutions[i].push(j);
            } else if dominates(&population[j].objectives, &population[i].objectives) {
                domination_count[i] += 1;
            }
        }

        if domination_count[i] == 0 {
            population[i].rank = 0;
            current_front.push(i);
        }
    }

    fronts.push(current_front.clone());

    // Subsequent fronts
    let mut rank = 0;
    while !fronts[rank].is_empty() {
        let mut next_front = Vec::new();

        for &i in &fronts[rank] {
            for &j in &dominated_solutions[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    population[j].rank = rank + 1;
                    next_front.push(j);
                }
            }
        }

        rank += 1;
        fronts.push(next_front.clone());
    }
}

/// Check if solution a dominates solution b
fn dominates<T: Float>(a: &[T], b: &[T]) -> bool {
    let mut better_in_any = false;

    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false; // a is worse in this objective
        }
        if ai < bi {
            better_in_any = true;
        }
    }

    better_in_any
}

/// Normalize objectives using ideal and nadir points
///
/// Performs objective normalization to ensure fair distance calculations:
/// 1. Translate by ideal point (minimum in each objective)
/// 2. Scale by range (nadir - ideal)
///
/// This ensures all objectives are on comparable scales before association.
///
/// # Arguments
///
/// * `population` - Population to normalize (modified in-place)
/// * `n_obj` - Number of objectives
///
/// # Degenerate Cases
///
/// When max = min for an objective (no variation), that objective is set to 0
/// after translation to avoid division by zero.
fn normalize_objectives<T: Float>(population: &mut [Individual<T>], n_obj: usize) -> Result<()> {
    if population.is_empty() {
        return Ok(());
    }

    // Find ideal point (minimum in each objective)
    let ideal = compute_ideal_point(population, n_obj);

    // Find nadir point (maximum in each objective)
    let nadir = compute_nadir_point(population, n_obj);

    // Normalize: (objective - ideal) / (nadir - ideal)
    for ind in population.iter_mut() {
        for i in 0..n_obj {
            let range = nadir[i] - ideal[i];

            // Handle degenerate case where all solutions have same value
            if range > T::epsilon() {
                let normalized = (ind.objectives[i] - ideal[i]) / range;
                ind.objectives[i] = normalized;
            } else {
                // No variation in this objective - set to zero after translation
                ind.objectives[i] = T::zero();
            }
        }
    }

    Ok(())
}

/// Compute ideal point (minimum in each objective)
fn compute_ideal_point<T: Float>(population: &[Individual<T>], n_obj: usize) -> Vec<T> {
    let mut ideal = vec![T::infinity(); n_obj];

    for ind in population.iter() {
        for (i, &obj) in ind.objectives.iter().enumerate() {
            if obj < ideal[i] {
                ideal[i] = obj;
            }
        }
    }

    ideal
}

/// Compute nadir point (maximum in each objective)
///
/// Note: In proper NSGA-III, the nadir point is estimated using extreme solutions
/// and achievement scalarizing function (ASF). This implementation uses a simpler
/// max-based approach which works well in practice.
fn compute_nadir_point<T: Float>(population: &[Individual<T>], n_obj: usize) -> Vec<T> {
    let mut nadir = vec![T::neg_infinity(); n_obj];

    for ind in population.iter() {
        for (i, &obj) in ind.objectives.iter().enumerate() {
            if obj > nadir[i] {
                nadir[i] = obj;
            }
        }
    }

    nadir
}

/// Associate solutions with nearest reference points using perpendicular distance
///
/// For each solution, finds the nearest reference line and calculates the
/// perpendicular distance from the solution to that line. The reference line
/// passes through the origin and the reference point direction.
///
/// This is a key feature of NSGA-III: using perpendicular distance ensures
/// solutions are associated with the closest reference direction, not just
/// the closest reference point.
fn associate_with_reference_points<T: Float + std::iter::Sum>(
    population: &mut [Individual<T>],
    reference_points: &[ReferencePoint<T>],
) -> Result<()> {
    for ind in population.iter_mut() {
        let (closest_idx, perp_dist) =
            find_closest_reference_point(&ind.objectives, reference_points)?;

        ind.reference_point_index = Some(closest_idx);
        ind.perpendicular_distance = perp_dist;
    }

    Ok(())
}

/// Find the closest reference point and perpendicular distance
///
/// # Arguments
///
/// * `objectives` - Objective values of the solution (normalized)
/// * `reference_points` - List of reference points (directions)
///
/// # Returns
///
/// Tuple of (reference_point_index, perpendicular_distance)
fn find_closest_reference_point<T: Float + std::iter::Sum>(
    objectives: &[T],
    reference_points: &[ReferencePoint<T>],
) -> Result<(usize, T)> {
    if reference_points.is_empty() {
        return Err(NumRs2Error::ValueError(
            "Reference points cannot be empty".to_string(),
        ));
    }

    let mut min_distance = T::infinity();
    let mut closest_idx = 0;

    for (idx, ref_point) in reference_points.iter().enumerate() {
        let dist = perpendicular_distance(objectives, &ref_point.position)?;

        if dist < min_distance {
            min_distance = dist;
            closest_idx = idx;
        }
    }

    Ok((closest_idx, min_distance))
}

/// Calculate perpendicular distance from a point to a reference line
///
/// The reference line passes through the origin with direction given by
/// the reference point. The perpendicular distance is:
///
/// d_perp = ||point - projection||
///
/// where projection = (point . ref_dir) * ref_dir
///
/// # Arguments
///
/// * `point` - Point in objective space (normalized)
/// * `ref_direction` - Reference direction (should be normalized)
///
/// # Returns
///
/// Perpendicular distance as T
///
/// # Mathematical Details
///
/// For a point p and reference direction r (unit vector):
/// 1. Project p onto r: proj = (p . r) * r
/// 2. Calculate perpendicular component: perp = p - proj
/// 3. Distance: d = ||perp||
fn perpendicular_distance<T: Float + std::iter::Sum>(
    point: &[T],
    ref_direction: &[T],
) -> Result<T> {
    if point.len() != ref_direction.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "Point and reference direction must have same dimension".to_string(),
        ));
    }

    if point.is_empty() {
        return Err(NumRs2Error::ValueError(
            "Cannot calculate distance for empty vectors".to_string(),
        ));
    }

    // Calculate dot product: point . ref_direction
    let dot_product: T = point
        .iter()
        .zip(ref_direction.iter())
        .map(|(&p, &r)| p * r)
        .sum();

    // Calculate projection: (point . ref_direction) * ref_direction
    let projection: Vec<T> = ref_direction.iter().map(|&r| dot_product * r).collect();

    // Calculate perpendicular component: point - projection
    let perpendicular: Vec<T> = point
        .iter()
        .zip(projection.iter())
        .map(|(&p, &proj)| p - proj)
        .collect();

    // Calculate L2 norm of perpendicular component
    let sum_squares: T = perpendicular.iter().map(|&x| x * x).sum();
    Ok(sum_squares.sqrt())
}

/// Calculate projection of a point onto a reference direction
///
/// Returns the scalar projection coefficient such that:
/// projection_vector = coefficient * ref_direction
fn scalar_projection<T: Float + std::iter::Sum>(point: &[T], ref_direction: &[T]) -> Result<T> {
    if point.len() != ref_direction.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "Dimension mismatch in scalar projection".to_string(),
        ));
    }

    // Calculate dot product: point . ref_direction
    // Assuming ref_direction is already normalized, this gives the projection coefficient
    let projection_coeff: T = point
        .iter()
        .zip(ref_direction.iter())
        .map(|(&p, &r)| p * r)
        .sum();

    Ok(projection_coeff)
}

/// Calculate Euclidean distance between two points
fn euclidean_distance<T: Float + std::iter::Sum>(a: &[T], b: &[T]) -> Result<T> {
    if a.len() != b.len() {
        return Err(NumRs2Error::DimensionMismatch(
            "Vectors must have same length".to_string(),
        ));
    }

    let sum_squares: T = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum();

    Ok(sum_squares.sqrt())
}

/// Niche preservation selection
///
/// Selects K individuals from the last front using reference point-based niching.
/// This is the core selection mechanism in NSGA-III that maintains diversity.
///
/// # Algorithm
///
/// 1. Initialize niche counts based on already selected individuals
/// 2. For each position to fill:
///    a. Find reference point(s) with minimum niche count
///    b. Among candidates associated with these reference points,
///    select the one with minimum perpendicular distance
///    c. Update niche counts
///
/// # Arguments
///
/// * `candidates` - Individuals from last front to select from
/// * `reference_points` - Reference points with current niche counts
/// * `k` - Number of individuals to select
///
/// # Returns
///
/// Vector of selected individuals
fn niche_preservation_selection<T: Float>(
    candidates: Vec<Individual<T>>,
    reference_points: &[ReferencePoint<T>],
    k: usize,
) -> Result<Vec<Individual<T>>> {
    if k == 0 {
        return Ok(Vec::new());
    }

    if candidates.is_empty() {
        return Err(NumRs2Error::ValueError(
            "No candidates available for selection".to_string(),
        ));
    }

    // Create mutable copy of reference points to track niche counts
    let mut ref_points_copy: Vec<ReferencePoint<T>> = reference_points.to_vec();

    // Track which candidates have been selected
    let mut selected_indices = Vec::new();
    let mut remaining_indices: Vec<usize> = (0..candidates.len()).collect();

    // Select K individuals
    for _ in 0..k.min(candidates.len()) {
        if remaining_indices.is_empty() {
            break;
        }

        // Find reference point(s) with minimum niche count
        let min_niche_count = ref_points_copy
            .iter()
            .map(|rp| rp.niche_count)
            .min()
            .ok_or_else(|| {
                NumRs2Error::ComputationError("No reference points available".to_string())
            })?;

        // Find all reference points with minimum niche count
        let min_niche_refs: Vec<usize> = ref_points_copy
            .iter()
            .enumerate()
            .filter(|(_, rp)| rp.niche_count == min_niche_count)
            .map(|(idx, _)| idx)
            .collect();

        // Find candidates associated with these reference points
        let mut candidates_in_min_niches: Vec<(usize, T)> = Vec::new();

        for &candidate_idx in &remaining_indices {
            let candidate = &candidates[candidate_idx];

            if let Some(ref_idx) = candidate.reference_point_index {
                if min_niche_refs.contains(&ref_idx) {
                    candidates_in_min_niches
                        .push((candidate_idx, candidate.perpendicular_distance));
                }
            }
        }

        // If no candidates in minimum niches, select from any remaining
        if candidates_in_min_niches.is_empty() {
            // This can happen if all candidates are associated with already-full niches
            // Select the one with smallest perpendicular distance overall
            let best_idx = *remaining_indices
                .iter()
                .min_by(|&&a, &&b| {
                    candidates[a]
                        .perpendicular_distance
                        .partial_cmp(&candidates[b].perpendicular_distance)
                        .unwrap_or(Ordering::Equal)
                })
                .ok_or_else(|| {
                    NumRs2Error::ComputationError("No candidates available".to_string())
                })?;

            selected_indices.push(best_idx);
            remaining_indices.retain(|&idx| idx != best_idx);

            // Update niche count
            if let Some(ref_idx) = candidates[best_idx].reference_point_index {
                ref_points_copy[ref_idx].niche_count += 1;
            }
        } else {
            // Select candidate with minimum perpendicular distance from minimum niches
            candidates_in_min_niches
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let (best_idx, _) = candidates_in_min_niches[0];
            selected_indices.push(best_idx);
            remaining_indices.retain(|&idx| idx != best_idx);

            // Update niche count for the associated reference point
            if let Some(ref_idx) = candidates[best_idx].reference_point_index {
                ref_points_copy[ref_idx].niche_count += 1;
            }
        }
    }

    // Collect selected individuals
    let selected: Vec<Individual<T>> = selected_indices
        .into_iter()
        .map(|idx| candidates[idx].clone())
        .collect();

    Ok(selected)
}

/// Update niche counts for reference points based on population
///
/// Counts how many individuals are associated with each reference point.
/// This is used to maintain diversity in the population.
///
/// # Arguments
///
/// * `population` - Current population with reference point associations
/// * `reference_points` - Reference points to update
fn update_niche_counts<T: Float>(
    population: &[Individual<T>],
    reference_points: &mut [ReferencePoint<T>],
) {
    // Reset all niche counts
    for ref_point in reference_points.iter_mut() {
        ref_point.niche_count = 0;
    }

    // Count individuals in each niche
    for individual in population {
        if let Some(ref_idx) = individual.reference_point_index {
            if ref_idx < reference_points.len() {
                reference_points[ref_idx].niche_count += 1;
            }
        }
    }
}

/// Binary tournament selection
fn tournament_selection<'a, T: Float>(
    population: &'a [Individual<T>],
    rng: &mut impl Rng,
) -> Result<&'a Individual<T>> {
    let n = population.len();

    let i1 = (rng.gen::<f64>() * n as f64) as usize % n;
    let i2 = (rng.gen::<f64>() * n as f64) as usize % n;

    if compare_individuals(&population[i1], &population[i2]) == Ordering::Less {
        Ok(&population[i1])
    } else {
        Ok(&population[i2])
    }
}

/// Compare individuals by rank and perpendicular distance
fn compare_individuals<T: Float>(a: &Individual<T>, b: &Individual<T>) -> Ordering {
    if a.rank < b.rank {
        Ordering::Less
    } else if a.rank > b.rank {
        Ordering::Greater
    } else if a.perpendicular_distance < b.perpendicular_distance {
        Ordering::Less
    } else if a.perpendicular_distance > b.perpendicular_distance {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Simulated Binary Crossover (SBX) - same as NSGA-II
fn sbx_crossover<T: Float>(
    parent1: &[T],
    parent2: &[T],
    bounds: &[(T, T)],
    eta: T,
    rng: &mut impl Rng,
) -> Result<(Vec<T>, Vec<T>)> {
    let n = parent1.len();
    let mut child1 = Vec::with_capacity(n);
    let mut child2 = Vec::with_capacity(n);

    for i in 0..n {
        let (lower, upper) = bounds[i];
        let p1 = parent1[i];
        let p2 = parent2[i];

        let rand_val = T::from(rng.gen::<f64>()).ok_or_else(|| {
            NumRs2Error::ConversionError("Random value conversion failed".to_string())
        })?;

        if (p1 - p2).abs()
            > T::from(1e-14).ok_or_else(|| {
                NumRs2Error::ConversionError("Epsilon conversion failed".to_string())
            })?
        {
            let (c1, c2) = if p1 < p2 { (p1, p2) } else { (p2, p1) };

            let beta = T::one()
                + (T::from(2.0).ok_or_else(|| {
                    NumRs2Error::ConversionError("Constant conversion failed".to_string())
                })? * (c1 - lower))
                    / (c2 - c1);
            let alpha = T::from(2.0).ok_or_else(|| {
                NumRs2Error::ConversionError("Constant conversion failed".to_string())
            })? - beta.powf(-(eta + T::one()));

            let beta_q = if rand_val <= (T::one() / alpha) {
                (rand_val * alpha).powf(T::one() / (eta + T::one()))
            } else {
                (T::one()
                    / (T::from(2.0).ok_or_else(|| {
                        NumRs2Error::ConversionError("Constant conversion failed".to_string())
                    })? - rand_val * alpha))
                    .powf(T::one() / (eta + T::one()))
            };

            let offspring1 = T::from(0.5).ok_or_else(|| {
                NumRs2Error::ConversionError("Constant conversion failed".to_string())
            })? * ((c1 + c2) - beta_q * (c2 - c1));
            let offspring2 = T::from(0.5).ok_or_else(|| {
                NumRs2Error::ConversionError("Constant conversion failed".to_string())
            })? * ((c1 + c2) + beta_q * (c2 - c1));

            child1.push(offspring1.max(lower).min(upper));
            child2.push(offspring2.max(lower).min(upper));
        } else {
            child1.push(p1);
            child2.push(p2);
        }
    }

    Ok((child1, child2))
}

/// Polynomial mutation - same as NSGA-II
fn polynomial_mutation<T: Float>(
    individual: &mut [T],
    bounds: &[(T, T)],
    eta: T,
    rng: &mut impl Rng,
) -> Result<()> {
    let n = individual.len();

    for i in 0..n {
        let (lower, upper) = bounds[i];
        let x = individual[i];

        let rand_val = T::from(rng.gen::<f64>()).ok_or_else(|| {
            NumRs2Error::ConversionError("Random value conversion failed".to_string())
        })?;

        let delta1 = (x - lower) / (upper - lower);
        let delta2 = (upper - x) / (upper - lower);

        let mut_pow = T::one() / (eta + T::one());

        let delta_q = if rand_val
            < T::from(0.5).ok_or_else(|| {
                NumRs2Error::ConversionError("Constant conversion failed".to_string())
            })? {
            let xy = T::one() - delta1;
            let val = T::from(2.0).ok_or_else(|| {
                NumRs2Error::ConversionError("Constant conversion failed".to_string())
            })? * rand_val
                + (T::one()
                    - T::from(2.0).ok_or_else(|| {
                        NumRs2Error::ConversionError("Constant conversion failed".to_string())
                    })? * rand_val)
                    * xy.powf(eta + T::one());
            val.powf(mut_pow) - T::one()
        } else {
            let xy = T::one() - delta2;
            let val = T::from(2.0).ok_or_else(|| {
                NumRs2Error::ConversionError("Constant conversion failed".to_string())
            })? * (T::one() - rand_val)
                + T::from(2.0).ok_or_else(|| {
                    NumRs2Error::ConversionError("Constant conversion failed".to_string())
                })? * (rand_val
                    - T::from(0.5).ok_or_else(|| {
                        NumRs2Error::ConversionError("Constant conversion failed".to_string())
                    })?)
                    * xy.powf(eta + T::one());
            T::one() - val.powf(mut_pow)
        };

        individual[i] = (x + delta_q * (upper - lower)).max(lower).min(upper);
    }

    Ok(())
}
