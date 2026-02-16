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

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use scirs2_core::random::{thread_rng, Distribution, Rng, Uniform};
use std::cmp::Ordering;

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

/// Crowding distance assignment
fn crowding_distance_assignment<T: Float>(population: &mut [Individual<T>], n_obj: usize) {
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

// =============================================================================
// Convergence Metrics
// =============================================================================

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
        .fold(
            T::infinity(),
            |min_dist, dist| {
                if dist < min_dist {
                    dist
                } else {
                    min_dist
                }
            },
        )
}

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
/// - Typical range: [0, ∞)
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
/// - Typical range: [0, ∞)
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

    let p_val = p.unwrap_or_else(|| T::from(2.0).expect("Default p=2.0 should convert to Float"));

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
fn euclidean_distance<T: Float + std::iter::Sum>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (*ai - *bi) * (*ai - *bi))
        .sum::<T>()
        .sqrt()
}

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
pub fn calculate_spacing<T: Float + std::fmt::Display + std::iter::Sum>(
    front: &[Vec<T>],
) -> Result<T> {
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
/// Spread (Δ) measures both the extent of spread and distribution uniformity.
/// Lower values indicate better spread and uniformity.
///
/// # Formula
///
/// Δ = (d_f + d_l + sum|d_i - d_mean|) / (d_f + d_l + (n-1)*d_mean)
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
/// - Δ = 0: Perfect spread and uniformity
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
    let denominator = d_f
        + d_l
        + d_mean
            * T::from(n - 1).ok_or_else(|| {
                NumRs2Error::ConversionError("Failed to convert n-1 to Float".to_string())
            })?;

    if denominator == T::zero() {
        return Ok(T::zero());
    }

    Ok(numerator / denominator)
}

// =============================================================================
// Pareto Frontier Validation Functions
// =============================================================================

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

#[cfg(test)]
#[path = "nsga2_tests.rs"]
mod tests;
