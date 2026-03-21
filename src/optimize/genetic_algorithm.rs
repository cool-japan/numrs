//! Genetic Algorithm optimization
//!
//! Genetic algorithms (GA) are population-based metaheuristic optimization algorithms
//! inspired by natural selection and genetics. They maintain a population of candidate
//! solutions and evolve them through selection, crossover, and mutation operations.
//!
//! # Features
//!
//! - Multiple selection strategies: Tournament, Roulette Wheel, Rank-based
//! - Various crossover operators: Single-point, Two-point, Uniform
//! - Mutation strategies: Gaussian, Uniform, Adaptive
//! - Elitism to preserve best solutions
//! - Configurable population size and genetic operators
//!
//! # Example
//!
//! ```
//! use numrs2::optimize::{genetic_algorithm, GAConfig, SelectionStrategy};
//!
//! // Minimize Rosenbrock function
//! let f = |x: &[f64]| {
//!     let (x0, x1) = (x[0], x[1]);
//!     (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
//! };
//!
//! let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
//! let config = GAConfig::default();
//!
//! let result = genetic_algorithm(f, &bounds, Some(config))
//!     .expect("GA should succeed");
//! ```

use crate::error::{NumRs2Error, Result};
use crate::optimize::{compute_norm, OptimizeResult};
use num_traits::Float;
use scirs2_core::random::{thread_rng, Distribution, Rng, RngExt, Uniform};
use std::cmp::Ordering;

/// Selection strategy for choosing parents
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionStrategy {
    /// Tournament selection with specified tournament size
    Tournament(usize),
    /// Roulette wheel (fitness-proportionate) selection
    RouletteWheel,
    /// Rank-based selection
    RankBased,
}

/// Crossover operator type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossoverType {
    /// Single-point crossover
    SinglePoint,
    /// Two-point crossover
    TwoPoint,
    /// Uniform crossover with specified probability
    Uniform(f64),
}

/// Mutation strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MutationStrategy {
    /// Gaussian mutation with specified standard deviation
    Gaussian(f64),
    /// Uniform mutation within bounds
    Uniform,
    /// Adaptive mutation that decreases over generations
    Adaptive,
}

/// Configuration for Genetic Algorithm
#[derive(Debug, Clone)]
pub struct GAConfig<T: Float> {
    /// Population size
    pub pop_size: usize,
    /// Number of generations
    pub max_generations: usize,
    /// Selection strategy
    pub selection: SelectionStrategy,
    /// Crossover type
    pub crossover: CrossoverType,
    /// Crossover probability
    pub crossover_rate: T,
    /// Mutation strategy
    pub mutation: MutationStrategy,
    /// Mutation probability
    pub mutation_rate: T,
    /// Elitism: number of best individuals to preserve
    pub elitism_count: usize,
    /// Convergence tolerance
    pub ftol: T,
}

impl<T: Float> Default for GAConfig<T> {
    fn default() -> Self {
        Self {
            pop_size: 100,
            max_generations: 200,
            selection: SelectionStrategy::Tournament(3),
            crossover: CrossoverType::TwoPoint,
            crossover_rate: T::from(0.8).expect("0.8 should convert to Float"),
            mutation: MutationStrategy::Gaussian(0.1),
            mutation_rate: T::from(0.1).expect("0.1 should convert to Float"),
            elitism_count: 2,
            ftol: T::from(1e-8).expect("1e-8 should convert to Float"),
        }
    }
}

/// Individual in the population
#[derive(Clone)]
struct Individual<T: Float> {
    chromosome: Vec<T>,
    fitness: T,
}

/// Genetic algorithm optimization
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `bounds` - Parameter bounds as (lower, upper) tuples
/// * `config` - Optional GA configuration
///
/// # Returns
///
/// `OptimizeResult` containing the best solution found
pub fn genetic_algorithm<T, F>(
    f: F,
    bounds: &[(T, T)],
    config: Option<GAConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Display + std::iter::Sum,
    F: Fn(&[T]) -> T,
{
    let config = config.unwrap_or_default();
    let dim = bounds.len();

    if dim == 0 {
        return Err(NumRs2Error::ValueError(
            "Bounds must have at least one dimension".to_string(),
        ));
    }

    if config.pop_size < 2 {
        return Err(NumRs2Error::ValueError(
            "Population size must be at least 2".to_string(),
        ));
    }

    let mut rng = thread_rng();

    // Initialize population
    let mut population = initialize_population::<T, F>(&f, bounds, config.pop_size, &mut rng)?;

    // Sort by fitness (ascending for minimization)
    population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));

    let mut best = population[0].clone();
    let mut nfev = config.pop_size;
    let njev = 0;
    let mut stagnation_count = 0;
    let max_stagnation = 20;

    for generation in 0..config.max_generations {
        let prev_best_fitness = best.fitness;

        // Elitism: preserve best individuals
        let mut new_population = Vec::with_capacity(config.pop_size);
        for i in 0..config.elitism_count.min(config.pop_size) {
            new_population.push(population[i].clone());
        }

        // Generate offspring
        while new_population.len() < config.pop_size {
            // Selection
            let parent1 = select_individual(&population, &config.selection, &mut rng)?;
            let parent2 = select_individual(&population, &config.selection, &mut rng)?;

            // Crossover
            let mut offspring = if T::from(rng.random::<f64>()).ok_or_else(|| {
                NumRs2Error::ConversionError("Float conversion failed".to_string())
            })? < config.crossover_rate
            {
                crossover(
                    &parent1.chromosome,
                    &parent2.chromosome,
                    &config.crossover,
                    &mut rng,
                )?
            } else {
                parent1.chromosome.clone()
            };

            // Mutation
            if T::from(rng.random::<f64>()).ok_or_else(|| {
                NumRs2Error::ConversionError("Float conversion failed".to_string())
            })? < config.mutation_rate
            {
                let mutation_strength = match config.mutation {
                    MutationStrategy::Adaptive => {
                        // Decrease mutation strength over generations
                        let progress = T::from(generation).ok_or_else(|| {
                            NumRs2Error::ConversionError("Generation conversion failed".to_string())
                        })? / T::from(config.max_generations).ok_or_else(|| {
                            NumRs2Error::ConversionError(
                                "Max generation conversion failed".to_string(),
                            )
                        })?;
                        T::from(0.5).ok_or_else(|| {
                            NumRs2Error::ConversionError("Constant conversion failed".to_string())
                        })? * (T::one() - progress)
                    }
                    MutationStrategy::Gaussian(sigma) => T::from(sigma).ok_or_else(|| {
                        NumRs2Error::ConversionError("Sigma conversion failed".to_string())
                    })?,
                    MutationStrategy::Uniform => T::zero(),
                };

                mutate(
                    &mut offspring,
                    bounds,
                    &config.mutation,
                    mutation_strength,
                    &mut rng,
                )?;
            }

            // Evaluate offspring
            let fitness = f(&offspring);
            nfev += 1;

            new_population.push(Individual {
                chromosome: offspring,
                fitness,
            });
        }

        // Replace population
        population = new_population;
        population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal));

        // Update best solution
        if population[0].fitness < best.fitness {
            best = population[0].clone();
            stagnation_count = 0;
        } else {
            stagnation_count += 1;
        }

        // Check convergence
        let improvement = (prev_best_fitness - best.fitness).abs();
        if improvement < config.ftol && generation > 10 {
            return Ok(OptimizeResult {
                x: best.chromosome,
                fun: best.fitness,
                grad: vec![T::zero(); dim],
                nit: generation + 1,
                nfev,
                njev,
                success: true,
                message: "Convergence achieved".to_string(),
            });
        }

        // Early stopping on stagnation
        if stagnation_count >= max_stagnation {
            return Ok(OptimizeResult {
                x: best.chromosome,
                fun: best.fitness,
                grad: vec![T::zero(); dim],
                nit: generation + 1,
                nfev,
                njev,
                success: true,
                message: "Stopped due to stagnation".to_string(),
            });
        }
    }

    Ok(OptimizeResult {
        x: best.chromosome,
        fun: best.fitness,
        grad: vec![T::zero(); dim],
        nit: config.max_generations,
        nfev,
        njev,
        success: true,
        message: "Maximum generations reached".to_string(),
    })
}

/// Initialize population with random individuals
fn initialize_population<T, F>(
    f: &F,
    bounds: &[(T, T)],
    pop_size: usize,
    rng: &mut impl Rng,
) -> Result<Vec<Individual<T>>>
where
    T: Float + std::fmt::Display,
    F: Fn(&[T]) -> T,
{
    let dim = bounds.len();
    let mut population = Vec::with_capacity(pop_size);

    for _ in 0..pop_size {
        let mut chromosome = Vec::with_capacity(dim);

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

            chromosome.push(value);
        }

        let fitness = f(&chromosome);
        population.push(Individual {
            chromosome,
            fitness,
        });
    }

    Ok(population)
}

/// Select an individual from the population
fn select_individual<'a, T: Float + std::iter::Sum>(
    population: &'a [Individual<T>],
    strategy: &SelectionStrategy,
    rng: &mut impl Rng,
) -> Result<&'a Individual<T>> {
    match strategy {
        SelectionStrategy::Tournament(k) => tournament_selection(population, *k, rng),
        SelectionStrategy::RouletteWheel => roulette_wheel_selection(population, rng),
        SelectionStrategy::RankBased => rank_based_selection(population, rng),
    }
}

/// Tournament selection
fn tournament_selection<'a, T: Float>(
    population: &'a [Individual<T>],
    tournament_size: usize,
    rng: &mut impl Rng,
) -> Result<&'a Individual<T>> {
    let mut best_idx = (rng.random::<f64>() * population.len() as f64) as usize % population.len();
    let mut best_fitness = population[best_idx].fitness;

    for _ in 1..tournament_size {
        let idx = (rng.random::<f64>() * population.len() as f64) as usize % population.len();
        if population[idx].fitness < best_fitness {
            best_idx = idx;
            best_fitness = population[idx].fitness;
        }
    }

    Ok(&population[best_idx])
}

/// Roulette wheel selection (fitness-proportionate)
fn roulette_wheel_selection<'a, T: Float + std::iter::Sum>(
    population: &'a [Individual<T>],
    rng: &mut impl Rng,
) -> Result<&'a Individual<T>> {
    // Convert to maximization problem by inverting fitness
    let max_fitness = population
        .iter()
        .map(|ind| ind.fitness)
        .fold(T::neg_infinity(), |a, b| if a > b { a } else { b });

    let fitnesses: Vec<T> = population
        .iter()
        .map(|ind| max_fitness - ind.fitness + T::one())
        .collect();

    let total_fitness: T = fitnesses.iter().copied().sum();

    let rand_val = T::from(rng.random::<f64>()).ok_or_else(|| {
        NumRs2Error::ConversionError("Random value conversion failed".to_string())
    })?;
    let mut threshold = rand_val * total_fitness;

    for (i, &fitness) in fitnesses.iter().enumerate() {
        threshold = threshold - fitness;
        if threshold <= T::zero() {
            return Ok(&population[i]);
        }
    }

    Ok(&population[population.len() - 1])
}

/// Rank-based selection
fn rank_based_selection<'a, T: Float>(
    population: &'a [Individual<T>],
    rng: &mut impl Rng,
) -> Result<&'a Individual<T>> {
    let n = population.len();
    let total_rank = n * (n + 1) / 2;

    let rand_val = rng.random::<f64>();
    let threshold = rand_val * total_rank as f64;

    let mut cumulative = 0.0;
    for (i, _) in population.iter().enumerate() {
        cumulative += (n - i) as f64;
        if cumulative >= threshold {
            return Ok(&population[i]);
        }
    }

    Ok(&population[n - 1])
}

/// Perform crossover between two parents
fn crossover<T: Float>(
    parent1: &[T],
    parent2: &[T],
    crossover_type: &CrossoverType,
    rng: &mut impl Rng,
) -> Result<Vec<T>> {
    let n = parent1.len();

    match crossover_type {
        CrossoverType::SinglePoint => {
            let point = (rng.random::<f64>() * (n - 1) as f64) as usize + 1;
            let mut offspring = Vec::with_capacity(n);
            offspring.extend_from_slice(&parent1[..point]);
            offspring.extend_from_slice(&parent2[point..]);
            Ok(offspring)
        }
        CrossoverType::TwoPoint => {
            let mut point1 = (rng.random::<f64>() * n as f64) as usize;
            let mut point2 = (rng.random::<f64>() * n as f64) as usize;

            if point1 > point2 {
                std::mem::swap(&mut point1, &mut point2);
            }

            let mut offspring = Vec::with_capacity(n);
            offspring.extend_from_slice(&parent1[..point1]);
            offspring.extend_from_slice(&parent2[point1..point2]);
            offspring.extend_from_slice(&parent1[point2..]);
            Ok(offspring)
        }
        CrossoverType::Uniform(prob) => {
            let mut offspring = Vec::with_capacity(n);
            let threshold = T::from(*prob).ok_or_else(|| {
                NumRs2Error::ConversionError("Probability conversion failed".to_string())
            })?;

            for i in 0..n {
                let rand_val = T::from(rng.random::<f64>()).ok_or_else(|| {
                    NumRs2Error::ConversionError("Random value conversion failed".to_string())
                })?;

                if rand_val < threshold {
                    offspring.push(parent1[i]);
                } else {
                    offspring.push(parent2[i]);
                }
            }
            Ok(offspring)
        }
    }
}

/// Mutate an individual's chromosome
fn mutate<T: Float>(
    chromosome: &mut [T],
    bounds: &[(T, T)],
    strategy: &MutationStrategy,
    strength: T,
    rng: &mut impl Rng,
) -> Result<()> {
    match strategy {
        MutationStrategy::Gaussian(_) => {
            for (i, gene) in chromosome.iter_mut().enumerate() {
                let (lower, upper) = bounds[i];
                let range = upper - lower;

                // Box-Muller transform for Gaussian random
                let u1 = rng.random::<f64>();
                let u2 = rng.random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

                let perturbation = T::from(z).ok_or_else(|| {
                    NumRs2Error::ConversionError("Gaussian sample conversion failed".to_string())
                })? * strength
                    * range;

                *gene = (*gene + perturbation).max(lower).min(upper);
            }
        }
        MutationStrategy::Uniform | MutationStrategy::Adaptive => {
            for (i, gene) in chromosome.iter_mut().enumerate() {
                let (lower, upper) = bounds[i];
                let range = upper - lower;

                let rand_val = T::from(rng.random::<f64>()).ok_or_else(|| {
                    NumRs2Error::ConversionError("Random value conversion failed".to_string())
                })?;

                let perturbation = (rand_val
                    - T::from(0.5).ok_or_else(|| {
                        NumRs2Error::ConversionError("Constant conversion failed".to_string())
                    })?)
                    * range
                    * strength;

                *gene = (*gene + perturbation).max(lower).min(upper);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ga_sphere_function() {
        // Minimize f(x,y) = x^2 + y^2
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = GAConfig {
            pop_size: 50,
            max_generations: 100,
            ..Default::default()
        };

        let result = genetic_algorithm(f, &bounds, Some(config)).expect("GA should succeed");
        assert!(result.success);
        assert!(result.fun < 0.1, "Should find good solution");
        assert_relative_eq!(result.x[0], 0.0, epsilon = 0.5);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 0.5);
    }

    #[test]
    fn test_ga_rosenbrock() {
        let f = |x: &[f64]| {
            let (x0, x1) = (x[0], x[1]);
            (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
        };
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = GAConfig {
            pop_size: 100,
            max_generations: 200,
            elitism_count: 5,
            ..Default::default()
        };

        let result = genetic_algorithm(f, &bounds, Some(config)).expect("GA should succeed");
        assert!(result.success);
        // Rosenbrock is challenging for GA; relax tolerance for stochastic nature
        // Global minimum is at (1.0, 1.0) with value 0.0
        assert!(
            result.fun < 10.0,
            "Should find reasonable solution near optimum"
        );
    }

    #[test]
    fn test_ga_single_point_crossover() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = GAConfig {
            crossover: CrossoverType::SinglePoint,
            pop_size: 50,
            max_generations: 50,
            ..Default::default()
        };

        let result = genetic_algorithm(f, &bounds, Some(config)).expect("GA should succeed");
        assert!(result.success);
    }

    #[test]
    fn test_ga_uniform_crossover() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = GAConfig {
            crossover: CrossoverType::Uniform(0.5),
            pop_size: 50,
            max_generations: 50,
            ..Default::default()
        };

        let result = genetic_algorithm(f, &bounds, Some(config)).expect("GA should succeed");
        assert!(result.success);
    }

    #[test]
    fn test_ga_roulette_wheel_selection() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = GAConfig {
            selection: SelectionStrategy::RouletteWheel,
            pop_size: 50,
            max_generations: 50,
            ..Default::default()
        };

        let result = genetic_algorithm(f, &bounds, Some(config)).expect("GA should succeed");
        assert!(result.success);
    }

    #[test]
    fn test_ga_rank_based_selection() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = GAConfig {
            selection: SelectionStrategy::RankBased,
            pop_size: 50,
            max_generations: 50,
            ..Default::default()
        };

        let result = genetic_algorithm(f, &bounds, Some(config)).expect("GA should succeed");
        assert!(result.success);
    }

    #[test]
    fn test_ga_adaptive_mutation() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = GAConfig {
            mutation: MutationStrategy::Adaptive,
            pop_size: 50,
            max_generations: 50,
            ..Default::default()
        };

        let result = genetic_algorithm(f, &bounds, Some(config)).expect("GA should succeed");
        assert!(result.success);
    }

    #[test]
    fn test_ga_invalid_bounds() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds: Vec<(f64, f64)> = vec![];

        let result = genetic_algorithm(f, &bounds, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ga_invalid_pop_size() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0)];

        let config = GAConfig {
            pop_size: 1,
            ..Default::default()
        };

        let result = genetic_algorithm(f, &bounds, Some(config));
        assert!(result.is_err());
    }
}
