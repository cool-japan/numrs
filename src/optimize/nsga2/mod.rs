//! NSGA-II: Non-dominated Sorting Genetic Algorithm II
//!
//! NSGA-II is a multi-objective evolutionary algorithm that uses non-dominated sorting
//! and crowding distance to maintain diversity in the population. It's particularly
//! effective for finding Pareto-optimal fronts in multi-objective optimization problems.
//!
//! # Features
//!
//! ## Core Algorithm
//! - Non-dominated sorting for ranking solutions
//! - Crowding distance calculation for diversity preservation
//! - Binary tournament selection based on rank and crowding distance
//! - Simulated Binary Crossover (SBX) and polynomial mutation
//!
//! ## Quality Metrics
//! - **Hypervolume Indicator**: Volume of objective space dominated by Pareto front
//! - **Spacing (S)**: Uniformity of distribution in Pareto front
//! - **Spread (Δ)**: Extent and uniformity of spread
//! - **IGD (Inverted Generational Distance)**: Convergence and coverage
//! - **GD (Generational Distance)**: Convergence to reference front
//!
//! ## Pareto Front Analysis
//! - Pareto frontier extraction and validation
//! - Non-dominated solution filtering
//! - Multi-objective sorting and ranking
//!
//! # Quality Metrics Guide
//!
//! ## Diversity Metrics
//!
//! ### Spacing (S)
//! Measures the uniformity of distribution in the Pareto front.
//!
//! **Formula**: S = sqrt(1/(n-1) * sum((d_i - d_mean)^2))
//!
//! **Interpretation**:
//! - S = 0: Perfectly uniform distribution
//! - Lower values: Better uniformity
//! - Typical range: [0, ∞)
//!
//! ### Spread (Δ)
//! Measures both the extent of spread and distribution uniformity.
//!
//! **Formula**: Δ = (d_f + d_l + sum|d_i - d_mean|) / (d_f + d_l + (n-1)*d_mean)
//!
//! **Interpretation**:
//! - Δ = 0: Perfect spread and uniformity
//! - Lower values: Better spread
//! - Typical range: [0, ∞)
//!
//! ## Convergence Metrics
//!
//! ### IGD (Inverted Generational Distance)
//! Measures how well the obtained front covers the reference front.
//!
//! **Formula**: IGD = (1/|P_ref|) * sqrt(sum(d_i^2))
//!
//! **Interpretation**:
//! - IGD = 0: Perfect coverage of reference front
//! - Lower values: Better convergence and coverage
//! - Typical range: [0, ∞)
//!
//! ### GD (Generational Distance)
//! Measures convergence to the reference front.
//!
//! **Formula**: GD = (1/n)^(1/p) * (sum(d_i^p))^(1/p)
//!
//! **Interpretation**:
//! - GD = 0: Perfect convergence
//! - Lower values: Better convergence
//! - Typical range: [0, ∞)
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use numrs2::optimize::nsga2::{nsga2, NSGA2Config};
//!
//! // Minimize two objectives: f1(x) = x^2, f2(x) = (x-2)^2
//! let objectives = vec![
//!     |x: &[f64]| x[0] * x[0],
//!     |x: &[f64]| (x[0] - 2.0).powi(2),
//! ];
//!
//! let bounds = vec![(0.0, 3.0)];
//! let config = NSGA2Config::default();
//!
//! let result = nsga2(&objectives, &bounds, Some(config))
//!     .expect("NSGA-II should succeed");
//!
//! println!("Pareto front size: {}", result.pareto_front.len());
//! ```
//!
//! ## With Quality Metrics
//!
//! ```
//! use numrs2::optimize::nsga2::{nsga2, NSGA2Config, QualityMetricsConfig};
//!
//! let objectives = vec![
//!     |x: &[f64]| x[0] * x[0],
//!     |x: &[f64]| (x[0] - 2.0).powi(2),
//! ];
//!
//! let bounds = vec![(0.0, 3.0)];
//!
//! let config = NSGA2Config {
//!     pop_size: 100,
//!     max_generations: 100,
//!     quality_metrics_config: Some(QualityMetricsConfig {
//!         calculate_spacing: true,
//!         calculate_spread: true,
//!         reference_front: None,
//!     }),
//!     ..Default::default()
//! };
//!
//! let result = nsga2(&objectives, &bounds, Some(config))
//!     .expect("NSGA-II should succeed");
//!
//! if let Some(spacing) = result.spacing {
//!     println!("Spacing: {:.4}", spacing);
//! }
//! if let Some(spread) = result.spread {
//!     println!("Spread: {:.4}", spread);
//! }
//! ```
//!
//! ## With Reference Front (IGD/GD)
//!
//! ```
//! use numrs2::optimize::nsga2::{nsga2, NSGA2Config, QualityMetricsConfig};
//!
//! let objectives = vec![
//!     |x: &[f64]| x[0] * x[0],
//!     |x: &[f64]| (x[0] - 2.0).powi(2),
//! ];
//!
//! let bounds = vec![(0.0, 3.0)];
//!
//! // Generate reference Pareto front
//! let mut reference_front = Vec::new();
//! for i in 0..20 {
//!     let x = i as f64 * 0.1;
//!     reference_front.push(vec![x * x, (x - 2.0).powi(2)]);
//! }
//!
//! let config = NSGA2Config {
//!     pop_size: 100,
//!     max_generations: 100,
//!     quality_metrics_config: Some(QualityMetricsConfig {
//!         calculate_spacing: true,
//!         calculate_spread: true,
//!         reference_front: Some(reference_front),
//!     }),
//!     ..Default::default()
//! };
//!
//! let result = nsga2(&objectives, &bounds, Some(config))
//!     .expect("NSGA-II should succeed");
//!
//! if let Some(igd) = result.igd {
//!     println!("IGD: {:.6}", igd);
//! }
//! if let Some(gd) = result.gd {
//!     println!("GD: {:.6}", gd);
//! }
//! ```
//!
//! # References
//!
//! - Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
//! - Zitzler, E., et al. (2003). "Performance assessment of multiobjective optimizers"
//! - Schott, J. R. (1995). "Fault Tolerant Design Using Single and Multicriteria Genetic Algorithm Optimization"

pub mod hypervolume;
pub mod metrics;
pub mod pareto;

#[cfg(test)]
mod tests;

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use scirs2_core::random::{thread_rng, Distribution, Rng, Uniform};
use std::cmp::Ordering;

// Re-export from submodules for a flat public API
pub use hypervolume::calculate_hypervolume;
pub use metrics::{calculate_gd, calculate_igd, calculate_spacing, calculate_spread};
pub use pareto::{
    extract_front_objectives, extract_non_dominated, extract_pareto_front,
    filter_dominated_solutions, is_pareto_optimal, sort_front_by_objective, validate_pareto_front,
};

// =============================================================================
// Public Types
// =============================================================================

/// Configuration for quality metrics calculation
#[derive(Debug, Clone)]
pub struct QualityMetricsConfig<T: Float> {
    /// Calculate spacing metric (uniformity of distribution)
    pub calculate_spacing: bool,
    /// Calculate spread metric (extent and uniformity)
    pub calculate_spread: bool,
    /// Reference Pareto front for IGD/GD calculation
    /// If provided, IGD and GD will be calculated
    pub reference_front: Option<Vec<Vec<T>>>,
}

impl<T: Float> Default for QualityMetricsConfig<T> {
    fn default() -> Self {
        Self {
            calculate_spacing: false,
            calculate_spread: false,
            reference_front: None,
        }
    }
}

/// Configuration for NSGA-II
#[derive(Debug, Clone)]
pub struct NSGA2Config<T: Float> {
    /// Population size (should be even)
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
    /// Optional hypervolume configuration
    pub hypervolume_config: Option<HypervolumeConfig<T>>,
    /// Optional quality metrics configuration
    pub quality_metrics_config: Option<QualityMetricsConfig<T>>,
}

impl<T: Float> Default for NSGA2Config<T> {
    fn default() -> Self {
        Self {
            pop_size: 100,
            max_generations: 100,
            crossover_rate: T::from(0.9).expect("0.9 should convert to Float"),
            mutation_rate: T::from(0.1).expect("0.1 should convert to Float"),
            eta_c: T::from(20.0).expect("20.0 should convert to Float"),
            eta_m: T::from(20.0).expect("20.0 should convert to Float"),
            hypervolume_config: None,
            quality_metrics_config: None,
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
    /// Crowding distance
    pub crowding_distance: T,
}

/// Result of NSGA-II optimization
#[derive(Debug)]
pub struct NSGA2Result<T: Float> {
    /// Pareto-optimal solutions
    pub pareto_front: Vec<Individual<T>>,
    /// All final population
    pub population: Vec<Individual<T>>,
    /// Number of generations executed
    pub generations: usize,
    /// Hypervolume indicator (if reference point provided)
    pub hypervolume: Option<T>,
    /// Spacing metric (uniformity of distribution)
    pub spacing: Option<T>,
    /// Spread metric (extent and uniformity)
    pub spread: Option<T>,
    /// Inverted Generational Distance (convergence to reference front)
    pub igd: Option<T>,
    /// Generational Distance (convergence to reference front)
    pub gd: Option<T>,
}

/// Configuration for hypervolume calculation
#[derive(Debug, Clone)]
pub struct HypervolumeConfig<T: Float> {
    /// Reference point for hypervolume calculation
    /// Must weakly dominate all points in the Pareto front
    pub reference_point: Vec<T>,
}

// =============================================================================
// Core Algorithm
// =============================================================================

/// NSGA-II multi-objective optimization
///
/// # Arguments
///
/// * `objectives` - Vector of objective functions to minimize
/// * `bounds` - Parameter bounds as (lower, upper) tuples
/// * `config` - Optional NSGA-II configuration
///
/// # Returns
///
/// `NSGA2Result` containing Pareto-optimal solutions
pub fn nsga2<T, F>(
    objectives: &[F],
    bounds: &[(T, T)],
    config: Option<NSGA2Config<T>>,
) -> Result<NSGA2Result<T>>
where
    T: Float + std::fmt::Display + std::iter::Sum,
    F: Fn(&[T]) -> T,
{
    let config = config.unwrap_or_default();
    let n_obj = objectives.len();
    let n_var = bounds.len();

    if n_obj < 2 {
        return Err(NumRs2Error::ValueError(
            "NSGA-II requires at least 2 objectives".to_string(),
        ));
    }

    if n_var == 0 {
        return Err(NumRs2Error::ValueError(
            "Bounds must have at least one dimension".to_string(),
        ));
    }

    if config.pop_size < 4 || !config.pop_size.is_multiple_of(2) {
        return Err(NumRs2Error::ValueError(
            "Population size must be at least 4 and even".to_string(),
        ));
    }

    let mut rng = thread_rng();

    // Initialize population
    let mut population = initialize_population(objectives, bounds, config.pop_size, &mut rng)?;

    // Evaluate and rank initial population
    fast_non_dominated_sort(&mut population);
    crowding_distance_assignment(&mut population, n_obj);

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

        // Environmental selection: select best pop_size individuals
        fast_non_dominated_sort(&mut population);
        crowding_distance_assignment(&mut population, n_obj);

        // Sort by rank and crowding distance
        population.sort_by(|a, b| compare_individuals(a, b));

        // Keep only pop_size best individuals
        population.truncate(config.pop_size);
    }

    // Extract Pareto front (rank 0)
    let pareto_front: Vec<Individual<T>> = population
        .iter()
        .filter(|ind| ind.rank == 0)
        .cloned()
        .collect();

    // Calculate hypervolume if reference point is provided
    let hypervolume = if let Some(hv_config) = &config.hypervolume_config {
        let front_objectives: Vec<Vec<T>> = pareto_front
            .iter()
            .map(|ind| ind.objectives.clone())
            .collect();

        calculate_hypervolume(&front_objectives, &hv_config.reference_point).ok()
    } else {
        None
    };

    // Calculate quality metrics if requested
    let (spacing, spread, igd, gd) = if let Some(qm_config) = &config.quality_metrics_config {
        let front_objectives: Vec<Vec<T>> = pareto_front
            .iter()
            .map(|ind| ind.objectives.clone())
            .collect();

        let spacing_val = if qm_config.calculate_spacing && front_objectives.len() >= 2 {
            calculate_spacing(&front_objectives).ok()
        } else {
            None
        };

        let spread_val = if qm_config.calculate_spread && front_objectives.len() >= 2 {
            calculate_spread(&front_objectives, None).ok()
        } else {
            None
        };

        let (igd_val, gd_val) = if let Some(ref_front) = &qm_config.reference_front {
            let igd = calculate_igd(&front_objectives, ref_front).ok();
            let gd = calculate_gd(&front_objectives, ref_front, None).ok();
            (igd, gd)
        } else {
            (None, None)
        };

        (spacing_val, spread_val, igd_val, gd_val)
    } else {
        (None, None, None, None)
    };

    Ok(NSGA2Result {
        pareto_front,
        population,
        generations: config.max_generations,
        hypervolume,
        spacing,
        spread,
        igd,
        gd,
    })
}

// =============================================================================
// Internal Helpers (pub(crate) for submodule access)
// =============================================================================

/// Check if solution a dominates solution b
///
/// a dominates b iff a is no worse in all objectives and strictly better in at least one.
pub(crate) fn dominates<T: Float>(a: &[T], b: &[T]) -> bool {
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
        crowding_distance: T::zero(),
    }
}

/// Fast non-dominated sorting
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

/// Crowding distance assignment
pub(crate) fn crowding_distance_assignment<T: Float>(
    population: &mut [Individual<T>],
    n_obj: usize,
) {
    let n = population.len();

    // Initialize crowding distances
    for ind in population.iter_mut() {
        ind.crowding_distance = T::zero();
    }

    // For each objective
    for m in 0..n_obj {
        // Sort by objective m
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            population[a].objectives[m]
                .partial_cmp(&population[b].objectives[m])
                .unwrap_or(Ordering::Equal)
        });

        // Set boundary solutions to infinite distance
        population[indices[0]].crowding_distance = T::infinity();
        population[indices[n - 1]].crowding_distance = T::infinity();

        // Calculate range
        let obj_min = population[indices[0]].objectives[m];
        let obj_max = population[indices[n - 1]].objectives[m];
        let obj_range = obj_max - obj_min;

        if obj_range > T::zero() {
            for i in 1..(n - 1) {
                if !population[indices[i]].crowding_distance.is_infinite() {
                    let distance = (population[indices[i + 1]].objectives[m]
                        - population[indices[i - 1]].objectives[m])
                        / obj_range;
                    population[indices[i]].crowding_distance =
                        population[indices[i]].crowding_distance + distance;
                }
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

/// Compare individuals by rank and crowding distance
fn compare_individuals<T: Float>(a: &Individual<T>, b: &Individual<T>) -> Ordering {
    if a.rank < b.rank {
        Ordering::Less
    } else if a.rank > b.rank {
        Ordering::Greater
    } else if a.crowding_distance > b.crowding_distance {
        Ordering::Less
    } else if a.crowding_distance < b.crowding_distance {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Simulated Binary Crossover (SBX)
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

/// Polynomial mutation
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
