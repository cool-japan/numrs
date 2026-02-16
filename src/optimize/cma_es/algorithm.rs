//! Core CMA-ES algorithm functions: single run, IPOP restarts, and helpers.

use crate::error::{NumRs2Error, Result};

use super::config::CMAESConfig;
use super::state::CmaEsState;
use super::types::{CMAESResult, RngSource, TerminationReason};

/// Run the CMA-ES optimization algorithm.
///
/// Minimizes an objective function `f` starting from initial point `x0`
/// using the provided configuration. CMA-ES is particularly effective for
/// non-convex, ill-conditioned, and multi-modal optimization problems
/// in moderate dimensions (n < 200).
///
/// # Arguments
/// * `f` - Objective function to minimize. Takes a slice `&[f64]` and returns `f64`.
/// * `x0` - Initial point (starting guess).
/// * `config` - CMA-ES configuration. Use [`CMAESConfig::default`] for defaults.
///
/// # Returns
/// A [`CMAESResult`] containing the best solution found and convergence info.
///
/// # Errors
/// Returns an error if the dimension is zero, sigma is non-positive,
/// or if internal numerical issues arise (e.g., eigendecomposition failure).
///
/// # Example
/// ```
/// use numrs2::optimize::cma_es::{cma_es, CMAESConfig};
///
/// let f = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
/// let x0 = vec![5.0, 3.0, -2.0];
/// let config = CMAESConfig::new(3).with_sigma0(1.0);
/// let result = cma_es(f, &x0, config).expect("CMA-ES should succeed");
/// assert!(result.fun < 1e-6);
/// ```
pub fn cma_es<F>(f: F, x0: &[f64], config: CMAESConfig) -> Result<CMAESResult>
where
    F: Fn(&[f64]) -> f64,
{
    if config.enable_restarts {
        cma_es_ipop(&f, x0, &config)
    } else {
        cma_es_single_run(&f, x0, &config)
    }
}

/// Single run of CMA-ES (no restarts).
fn cma_es_single_run<F>(f: &F, x0: &[f64], config: &CMAESConfig) -> Result<CMAESResult>
where
    F: Fn(&[f64]) -> f64,
{
    let mut state = CmaEsState::new(x0, config)?;

    // Evaluate initial point
    let f0 = state.evaluate(f, x0, &config.bounds, config.penalty_coefficient);
    state.best_f = f0;
    state.best_x = x0.to_vec();
    state.history.push(f0);

    loop {
        // Sample population
        let mut population = state.sample_population();

        // Apply boundary repair if bounds are specified
        if let Some(ref bounds) = config.bounds {
            for x in &mut population {
                CmaEsState::repair_bounds(x, bounds);
            }
        }

        // Evaluate all candidates
        let fitness_values: Vec<f64> = population
            .iter()
            .map(|x| state.evaluate(f, x, &config.bounds, config.penalty_coefficient))
            .collect();

        // Sort population by fitness (ascending = minimization)
        let mut indices: Vec<usize> = (0..state.lambda).collect();
        indices.sort_by(|&a, &b| {
            fitness_values[a]
                .partial_cmp(&fitness_values[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_pop: Vec<Vec<f64>> = indices.iter().map(|&i| population[i].clone()).collect();
        let sorted_fitness: Vec<f64> = indices.iter().map(|&i| fitness_values[i]).collect();

        // Update best
        if sorted_fitness[0] < state.best_f {
            state.best_f = sorted_fitness[0];
            state.best_x = sorted_pop[0].clone();
        }
        state.history.push(state.best_f);

        // Check termination before updating
        if let Some(reason) = state.check_termination(config, &sorted_fitness) {
            let success = matches!(
                reason,
                TerminationReason::FunctionTolerance | TerminationReason::ParameterTolerance
            );
            let final_condition = state.condition_number();
            let history = state.history;
            return Ok(CMAESResult {
                x: state.best_x,
                fun: state.best_f,
                nit: state.generation,
                nfev: state.function_evaluations,
                success,
                message: format!("{}", reason),
                history,
                final_sigma: state.sigma,
                final_condition_number: final_condition,
                restarts: 0,
                termination_reason: reason,
            });
        }

        // Save old mean for step computation
        let old_mean = state.update_mean(&sorted_pop);

        // Compute weighted step y_w = (m_new - m_old) / sigma
        let y_w = state.compute_weighted_step(&old_mean);

        // Update step-size via CSA
        state.update_step_size(&y_w);

        // Update covariance matrix
        state.update_covariance(&y_w, &sorted_pop, &old_mean);

        // Update eigendecomposition
        state.update_eigendecomposition()?;

        state.generation += 1;
    }
}

/// IPOP-CMA-ES: CMA-ES with increasing population size restarts.
///
/// When a single CMA-ES run terminates without satisfying convergence criteria,
/// the algorithm restarts with an increased population size. This helps escape
/// local optima and converge on multi-modal problems.
fn cma_es_ipop<F>(f: &F, x0: &[f64], config: &CMAESConfig) -> Result<CMAESResult>
where
    F: Fn(&[f64]) -> f64,
{
    let mut best_result: Option<CMAESResult> = None;
    let mut total_fevals = 0;
    let mut total_restarts = 0;
    let mut current_lambda = config.effective_lambda(x0.len());

    // Create a derived seed for restart perturbation
    let mut restart_rng = RngSource::create(config.seed.map(|s| s.wrapping_add(42)));

    for restart in 0..=config.max_restarts {
        let mut run_config = config.clone();
        run_config.population_size = current_lambda;
        run_config.enable_restarts = false; // Avoid recursion

        // Advance seed for each restart to get different sampling
        if let Some(base_seed) = config.seed {
            run_config.seed = Some(base_seed.wrapping_add(restart as u64 * 1000));
        }

        // On restarts, perturb the starting point
        let start_point = if restart == 0 {
            x0.to_vec()
        } else {
            perturb_start_point(x0, config.sigma0, &config.bounds, &mut restart_rng)
        };

        let result = cma_es_single_run(f, &start_point, &run_config)?;
        total_fevals += result.nfev;

        let is_better = match &best_result {
            None => true,
            Some(prev) => result.fun < prev.fun,
        };

        if is_better {
            best_result = Some(result.clone());
        }

        // If converged, we're done
        if result.success {
            let mut final_result = best_result.ok_or_else(|| {
                NumRs2Error::ComputationError("No result available after convergence".to_string())
            })?;
            final_result.nfev = total_fevals;
            final_result.restarts = restart;
            return Ok(final_result);
        }

        total_restarts += 1;

        // Increase population size
        current_lambda = (current_lambda as f64 * config.restart_pop_multiplier).ceil() as usize;
    }

    // Return the best result found, marked as not converged
    match best_result {
        Some(mut result) => {
            result.nfev = total_fevals;
            result.restarts = total_restarts;
            result.termination_reason = TerminationReason::NoImprovementAfterRestarts;
            result.success = false;
            result.message = format!("{}", TerminationReason::NoImprovementAfterRestarts);
            Ok(result)
        }
        None => Err(NumRs2Error::ComputationError(
            "CMA-ES IPOP produced no results".to_string(),
        )),
    }
}

/// Perturb the starting point for restarts.
fn perturb_start_point(
    x0: &[f64],
    sigma: f64,
    bounds: &Option<Vec<(f64, f64)>>,
    rng: &mut RngSource,
) -> Vec<f64> {
    let mut point: Vec<f64> = x0
        .iter()
        .map(|&xi| xi + rng.sample_normal_with_std(sigma))
        .collect();

    // Clamp to bounds if specified
    if let Some(ref b) = bounds {
        for (i, &(lo, hi)) in b.iter().enumerate() {
            if i < point.len() {
                point[i] = point[i].clamp(lo, hi);
            }
        }
    }

    point
}
