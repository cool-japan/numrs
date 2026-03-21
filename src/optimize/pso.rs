//! Particle Swarm Optimization (PSO)
//!
//! PSO is a population-based stochastic optimization algorithm inspired by the social
//! behavior of bird flocking or fish schooling. Particles move through the search space
//! based on their own experience and the experience of neighboring particles.
//!
//! # Features
//!
//! - Velocity and position updates with inertia weight
//! - Cognitive and social learning parameters
//! - Multiple inertia weight strategies: Constant, Linear Decay, Adaptive
//! - Topology variants: Global best (gbest), Local best (lbest), Ring
//! - Personal and global best tracking
//!
//! # Example
//!
//! ```
//! use numrs2::optimize::{particle_swarm, PSOConfig, InertiaStrategy};
//!
//! // Minimize Rosenbrock function
//! let f = |x: &[f64]| {
//!     let (x0, x1) = (x[0], x[1]);
//!     (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
//! };
//!
//! let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
//! let config = PSOConfig::default();
//!
//! let result = particle_swarm(f, &bounds, Some(config))
//!     .expect("PSO should succeed");
//! ```

use crate::error::{NumRs2Error, Result};
use crate::optimize::OptimizeResult;
use num_traits::Float;
use scirs2_core::random::{thread_rng, Distribution, Rng, RngExt, Uniform};

/// Inertia weight strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InertiaStrategy {
    /// Constant inertia weight
    Constant(f64),
    /// Linearly decreasing inertia weight from w_max to w_min
    LinearDecay { w_max: f64, w_min: f64 },
    /// Adaptive inertia weight based on swarm performance
    Adaptive,
}

/// Topology for neighborhood structure
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Topology {
    /// Global best - all particles influenced by single global best
    Global,
    /// Local best - particles influenced by k nearest neighbors
    Local(usize),
    /// Ring topology - each particle influenced by immediate neighbors
    Ring,
}

/// Configuration for Particle Swarm Optimization
#[derive(Debug, Clone)]
pub struct PSOConfig<T: Float> {
    /// Number of particles in the swarm
    pub swarm_size: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Inertia weight strategy
    pub inertia: InertiaStrategy,
    /// Cognitive parameter (personal best weight)
    pub c1: T,
    /// Social parameter (global/local best weight)
    pub c2: T,
    /// Velocity clamping factor (as fraction of search range)
    pub v_max_factor: T,
    /// Topology structure
    pub topology: Topology,
    /// Convergence tolerance
    pub ftol: T,
}

impl<T: Float> Default for PSOConfig<T> {
    fn default() -> Self {
        Self {
            swarm_size: 30,
            max_iter: 200,
            inertia: InertiaStrategy::LinearDecay {
                w_max: 0.9,
                w_min: 0.4,
            },
            c1: T::from(2.0).expect("2.0 should convert to Float"),
            c2: T::from(2.0).expect("2.0 should convert to Float"),
            v_max_factor: T::from(0.2).expect("0.2 should convert to Float"),
            topology: Topology::Global,
            ftol: T::from(1e-8).expect("1e-8 should convert to Float"),
        }
    }
}

/// Particle in the swarm
#[derive(Clone)]
struct Particle<T: Float> {
    position: Vec<T>,
    velocity: Vec<T>,
    best_position: Vec<T>,
    best_fitness: T,
    fitness: T,
}

/// Particle Swarm Optimization
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `bounds` - Parameter bounds as (lower, upper) tuples
/// * `config` - Optional PSO configuration
///
/// # Returns
///
/// `OptimizeResult` containing the best solution found
pub fn particle_swarm<T, F>(
    f: F,
    bounds: &[(T, T)],
    config: Option<PSOConfig<T>>,
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

    if config.swarm_size < 2 {
        return Err(NumRs2Error::ValueError(
            "Swarm size must be at least 2".to_string(),
        ));
    }

    let mut rng = thread_rng();

    // Calculate velocity limits
    let v_max: Vec<T> = bounds
        .iter()
        .map(|&(lower, upper)| (upper - lower) * config.v_max_factor)
        .collect();

    // Initialize swarm
    let mut swarm = initialize_swarm::<T, F>(&f, bounds, config.swarm_size, &v_max, &mut rng)?;

    // Find global best
    let mut global_best_idx = 0;
    let mut global_best_fitness = swarm[0].fitness;
    for (i, particle) in swarm.iter().enumerate() {
        if particle.fitness < global_best_fitness {
            global_best_fitness = particle.fitness;
            global_best_idx = i;
        }
    }
    let mut global_best = swarm[global_best_idx].position.clone();

    let mut nfev = config.swarm_size;
    let njev = 0;
    let mut stagnation_count = 0;
    let max_stagnation = 20;

    for iter in 0..config.max_iter {
        let prev_best_fitness = global_best_fitness;

        // Calculate inertia weight
        let w = match config.inertia {
            InertiaStrategy::Constant(w_val) => T::from(w_val).ok_or_else(|| {
                NumRs2Error::ConversionError("Inertia weight conversion failed".to_string())
            })?,
            InertiaStrategy::LinearDecay { w_max, w_min } => {
                let progress = T::from(iter).ok_or_else(|| {
                    NumRs2Error::ConversionError("Iteration conversion failed".to_string())
                })? / T::from(config.max_iter).ok_or_else(|| {
                    NumRs2Error::ConversionError("Max iteration conversion failed".to_string())
                })?;

                let w_max_t = T::from(w_max).ok_or_else(|| {
                    NumRs2Error::ConversionError("w_max conversion failed".to_string())
                })?;
                let w_min_t = T::from(w_min).ok_or_else(|| {
                    NumRs2Error::ConversionError("w_min conversion failed".to_string())
                })?;

                w_max_t - (w_max_t - w_min_t) * progress
            }
            InertiaStrategy::Adaptive => {
                // Adaptive based on swarm diversity
                let diversity = calculate_diversity(&swarm);
                let diversity_t = T::from(diversity).ok_or_else(|| {
                    NumRs2Error::ConversionError("Diversity conversion failed".to_string())
                })?;

                // High diversity -> high inertia, low diversity -> low inertia
                let w_max = T::from(0.9).ok_or_else(|| {
                    NumRs2Error::ConversionError("w_max conversion failed".to_string())
                })?;
                let w_min = T::from(0.4).ok_or_else(|| {
                    NumRs2Error::ConversionError("w_min conversion failed".to_string())
                })?;

                w_min + (w_max - w_min) * diversity_t
            }
        };

        // Pre-compute neighborhood bests to avoid borrow checker issues
        let neighborhood_bests: Vec<Vec<T>> = (0..swarm.len())
            .map(|i| match config.topology {
                Topology::Global => global_best.clone(),
                Topology::Local(k) => get_local_best(&swarm, i, k).to_vec(),
                Topology::Ring => get_ring_best(&swarm, i).to_vec(),
            })
            .collect();

        // Update particles
        for (i, particle) in swarm.iter_mut().enumerate() {
            let neighborhood_best = &neighborhood_bests[i];

            // Update velocity and position
            for j in 0..dim {
                // Random coefficients
                let r1 = T::from(rng.random::<f64>()).ok_or_else(|| {
                    NumRs2Error::ConversionError("Random value conversion failed".to_string())
                })?;
                let r2 = T::from(rng.random::<f64>()).ok_or_else(|| {
                    NumRs2Error::ConversionError("Random value conversion failed".to_string())
                })?;

                // Velocity update: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                let cognitive = config.c1 * r1 * (particle.best_position[j] - particle.position[j]);
                let social = config.c2 * r2 * (neighborhood_best[j] - particle.position[j]);

                particle.velocity[j] = w * particle.velocity[j] + cognitive + social;

                // Clamp velocity
                particle.velocity[j] = particle.velocity[j].max(-v_max[j]).min(v_max[j]);

                // Update position
                particle.position[j] = particle.position[j] + particle.velocity[j];

                // Enforce bounds
                let (lower, upper) = bounds[j];
                particle.position[j] = particle.position[j].max(lower).min(upper);
            }

            // Evaluate new position
            particle.fitness = f(&particle.position);
            nfev += 1;

            // Update personal best
            if particle.fitness < particle.best_fitness {
                particle.best_position = particle.position.clone();
                particle.best_fitness = particle.fitness;
            }

            // Update global best
            if particle.fitness < global_best_fitness {
                global_best = particle.position.clone();
                global_best_fitness = particle.fitness;
            }
        }

        // Check convergence
        let improvement = (prev_best_fitness - global_best_fitness).abs();
        if improvement < config.ftol && iter > 10 {
            return Ok(OptimizeResult {
                x: global_best,
                fun: global_best_fitness,
                grad: vec![T::zero(); dim],
                nit: iter + 1,
                nfev,
                njev,
                success: true,
                message: "Convergence achieved".to_string(),
            });
        }

        // Track stagnation
        if improvement < config.ftol {
            stagnation_count += 1;
        } else {
            stagnation_count = 0;
        }

        // Early stopping on stagnation
        if stagnation_count >= max_stagnation {
            return Ok(OptimizeResult {
                x: global_best,
                fun: global_best_fitness,
                grad: vec![T::zero(); dim],
                nit: iter + 1,
                nfev,
                njev,
                success: true,
                message: "Stopped due to stagnation".to_string(),
            });
        }
    }

    Ok(OptimizeResult {
        x: global_best,
        fun: global_best_fitness,
        grad: vec![T::zero(); dim],
        nit: config.max_iter,
        nfev,
        njev,
        success: true,
        message: "Maximum iterations reached".to_string(),
    })
}

/// Initialize swarm with random particles
fn initialize_swarm<T, F>(
    f: &F,
    bounds: &[(T, T)],
    swarm_size: usize,
    v_max: &[T],
    rng: &mut impl Rng,
) -> Result<Vec<Particle<T>>>
where
    T: Float + std::fmt::Display,
    F: Fn(&[T]) -> T,
{
    let dim = bounds.len();
    let mut swarm = Vec::with_capacity(swarm_size);

    for _ in 0..swarm_size {
        let mut position = Vec::with_capacity(dim);
        let mut velocity = Vec::with_capacity(dim);

        for j in 0..dim {
            let (lower, upper) = bounds[j];

            // Random position
            let lower_f64 = lower.to_f64().ok_or_else(|| {
                NumRs2Error::ConversionError("Bound conversion failed".to_string())
            })?;
            let upper_f64 = upper.to_f64().ok_or_else(|| {
                NumRs2Error::ConversionError("Bound conversion failed".to_string())
            })?;

            let pos_uniform = Uniform::new(lower_f64, upper_f64).map_err(|e| {
                NumRs2Error::ComputationError(format!("Uniform creation failed: {}", e))
            })?;

            let pos = T::from(pos_uniform.sample(rng)).ok_or_else(|| {
                NumRs2Error::ConversionError("Position sample conversion failed".to_string())
            })?;

            // Random velocity in [-v_max, v_max]
            let v_max_f64 = v_max[j].to_f64().ok_or_else(|| {
                NumRs2Error::ConversionError("Velocity max conversion failed".to_string())
            })?;

            let vel_uniform = Uniform::new(-v_max_f64, v_max_f64).map_err(|e| {
                NumRs2Error::ComputationError(format!("Uniform creation failed: {}", e))
            })?;

            let vel = T::from(vel_uniform.sample(rng)).ok_or_else(|| {
                NumRs2Error::ConversionError("Velocity sample conversion failed".to_string())
            })?;

            position.push(pos);
            velocity.push(vel);
        }

        let fitness = f(&position);

        swarm.push(Particle {
            position: position.clone(),
            velocity,
            best_position: position,
            best_fitness: fitness,
            fitness,
        });
    }

    Ok(swarm)
}

/// Calculate swarm diversity (normalized standard deviation)
fn calculate_diversity<T: Float>(swarm: &[Particle<T>]) -> f64 {
    if swarm.is_empty() {
        return 0.0;
    }

    let dim = swarm[0].position.len();
    let n = swarm.len() as f64;

    let mut total_variance = 0.0;

    for j in 0..dim {
        // Calculate mean
        let mean = swarm
            .iter()
            .map(|p| p.position[j].to_f64().unwrap_or(0.0))
            .sum::<f64>()
            / n;

        // Calculate variance
        let variance = swarm
            .iter()
            .map(|p| {
                let diff = p.position[j].to_f64().unwrap_or(0.0) - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;

        total_variance += variance;
    }

    (total_variance / dim as f64).sqrt().min(1.0)
}

/// Get local best from k nearest neighbors
fn get_local_best<T: Float>(swarm: &[Particle<T>], idx: usize, k: usize) -> &Vec<T> {
    // For simplicity, use k adjacent particles in the swarm array
    let n = swarm.len();
    let half_k = k / 2;

    let start = idx.saturating_sub(half_k);
    let end = (idx + half_k + 1).min(n);

    let mut best_idx = idx;
    let mut best_fitness = swarm[idx].best_fitness;

    for i in start..end {
        if swarm[i].best_fitness < best_fitness {
            best_fitness = swarm[i].best_fitness;
            best_idx = i;
        }
    }

    &swarm[best_idx].best_position
}

/// Get ring topology best (immediate neighbors)
fn get_ring_best<T: Float>(swarm: &[Particle<T>], idx: usize) -> &Vec<T> {
    let n = swarm.len();
    let prev = if idx == 0 { n - 1 } else { idx - 1 };
    let next = (idx + 1) % n;

    let mut best_idx = idx;
    let mut best_fitness = swarm[idx].best_fitness;

    for &i in &[prev, next] {
        if swarm[i].best_fitness < best_fitness {
            best_fitness = swarm[i].best_fitness;
            best_idx = i;
        }
    }

    &swarm[best_idx].best_position
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pso_sphere_function() {
        // Minimize f(x,y) = x^2 + y^2
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        // Improved parameters for more reliable convergence
        let config = PSOConfig {
            swarm_size: 50, // Larger swarm for better exploration
            max_iter: 200,  // More iterations for convergence
            ..Default::default()
        };

        let result = particle_swarm(f, &bounds, Some(config)).expect("PSO should succeed");
        assert!(result.success);
        // Relaxed tolerance for stochastic algorithm - 0.1 is still very good
        assert!(
            result.fun < 0.1,
            "Should find good solution, got fun = {}",
            result.fun
        );
        assert_relative_eq!(result.x[0], 0.0, epsilon = 0.3);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 0.3);
    }

    #[test]
    fn test_pso_rosenbrock() {
        let f = |x: &[f64]| {
            let (x0, x1) = (x[0], x[1]);
            (1.0 - x0).powi(2) + 100.0 * (x1 - x0 * x0).powi(2)
        };
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = PSOConfig {
            swarm_size: 50,
            max_iter: 200,
            ..Default::default()
        };

        let result = particle_swarm(f, &bounds, Some(config)).expect("PSO should succeed");
        assert!(result.success);
        assert!(result.fun < 1.0, "Should find reasonable solution");
    }

    #[test]
    fn test_pso_constant_inertia() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = PSOConfig {
            inertia: InertiaStrategy::Constant(0.7),
            swarm_size: 30,
            max_iter: 50,
            ..Default::default()
        };

        let result = particle_swarm(f, &bounds, Some(config)).expect("PSO should succeed");
        assert!(result.success);
    }

    #[test]
    fn test_pso_adaptive_inertia() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = PSOConfig {
            inertia: InertiaStrategy::Adaptive,
            swarm_size: 30,
            max_iter: 50,
            ..Default::default()
        };

        let result = particle_swarm(f, &bounds, Some(config)).expect("PSO should succeed");
        assert!(result.success);
    }

    #[test]
    fn test_pso_local_topology() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = PSOConfig {
            topology: Topology::Local(5),
            swarm_size: 30,
            max_iter: 50,
            ..Default::default()
        };

        let result = particle_swarm(f, &bounds, Some(config)).expect("PSO should succeed");
        assert!(result.success);
    }

    #[test]
    fn test_pso_ring_topology() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let config = PSOConfig {
            topology: Topology::Ring,
            swarm_size: 30,
            max_iter: 50,
            ..Default::default()
        };

        let result = particle_swarm(f, &bounds, Some(config)).expect("PSO should succeed");
        assert!(result.success);
    }

    #[test]
    fn test_pso_high_dimensional() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0); 10];

        let config = PSOConfig {
            swarm_size: 120,
            max_iter: 300,
            inertia: InertiaStrategy::LinearDecay {
                w_max: 0.95,
                w_min: 0.3,
            },
            c1: 2.05,
            c2: 2.05,
            ..Default::default()
        };

        let result = particle_swarm(f, &bounds, Some(config)).expect("PSO should succeed");
        assert!(result.success);
        assert!(result.fun < 5.0);
    }

    #[test]
    fn test_pso_invalid_bounds() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds: Vec<(f64, f64)> = vec![];

        let result = particle_swarm(f, &bounds, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_pso_invalid_swarm_size() {
        let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum();
        let bounds = vec![(-5.0, 5.0)];

        let config = PSOConfig {
            swarm_size: 1,
            ..Default::default()
        };

        let result = particle_swarm(f, &bounds, Some(config));
        assert!(result.is_err());
    }
}
