//! Derivative-free optimization algorithms
//!
//! This module provides optimization methods that do not require gradient information:
//! - Nelder-Mead simplex method

use crate::error::Result;
use num_traits::Float;

use super::{OptimizeConfig, OptimizeResult};

/// Nelder-Mead simplex optimization (derivative-free)
///
/// Minimizes a scalar function without requiring gradient information.
/// Useful for non-smooth or noisy functions.
///
/// # Arguments
///
/// * `f` - Objective function to minimize
/// * `x0` - Initial guess
/// * `config` - Optional configuration
///
/// # Examples
///
/// ```
/// use numrs2::optimize::*;
///
/// // Minimize a non-smooth function
/// let f = |x: &[f64]| (x[0] - 2.0).abs() + (x[1] + 3.0).abs();
///
/// let result = nelder_mead(f, &[0.0, 0.0], None).expect("Nelder-Mead optimization should succeed");
/// assert!(result.success);
/// ```
pub fn nelder_mead<T, F>(
    f: F,
    x0: &[T],
    config: Option<OptimizeConfig<T>>,
) -> Result<OptimizeResult<T>>
where
    T: Float + std::fmt::Debug + std::iter::Sum,
    F: Fn(&[T]) -> T,
{
    let cfg = config.unwrap_or_default();
    let n = x0.len();

    // Nelder-Mead coefficients
    let alpha = T::one(); // Reflection
    let gamma = T::from(2.0).expect("2.0 should be representable in Float"); // Expansion
    let rho = T::from(0.5).expect("0.5 should be representable in Float"); // Contraction
    let sigma = T::from(0.5).expect("0.5 should be representable in Float"); // Shrink

    // Initialize simplex (n+1 vertices)
    let mut simplex: Vec<Vec<T>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());

    // Create initial simplex using perturbations
    for i in 0..n {
        let mut vertex = x0.to_vec();
        vertex[i] = vertex[i] + T::from(0.05).expect("0.05 should be representable in Float");
        simplex.push(vertex);
    }

    // Evaluate function at all vertices
    let mut f_vals: Vec<T> = simplex.iter().map(|x| f(x)).collect();
    let mut nfev = n + 1;

    // Main iteration
    for iter in 0..cfg.max_iter {
        // Sort simplex by function values
        let mut indices: Vec<usize> = (0..n + 1).collect();
        indices.sort_by(|&i, &j| {
            f_vals[i]
                .partial_cmp(&f_vals[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Best, worst, and second-worst points
        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        // Check convergence: range of function values
        let f_range = f_vals[worst_idx] - f_vals[best_idx];
        if f_range < cfg.ftol {
            return Ok(OptimizeResult {
                x: simplex[best_idx].clone(),
                fun: f_vals[best_idx],
                grad: vec![T::zero(); n], // No gradient in derivative-free method
                nit: iter,
                nfev,
                njev: 0,
                success: true,
                message: "Optimization terminated successfully (simplex converged)".to_string(),
            });
        }

        // Compute centroid of all points except worst
        let mut centroid = vec![T::zero(); n];
        for &idx in indices.iter().take(n) {
            for j in 0..n {
                centroid[j] = centroid[j] + simplex[idx][j];
            }
        }
        for j in 0..n {
            centroid[j] = centroid[j] / T::from(n).expect("n should be representable in Float");
        }

        // Reflection: x_r = centroid + alpha * (centroid - x_worst)
        let x_r: Vec<T> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]))
            .collect();
        let f_r = f(&x_r);
        nfev += 1;

        if f_r < f_vals[best_idx] {
            // Expansion: try going further
            let x_e: Vec<T> = (0..n)
                .map(|j| centroid[j] + gamma * (x_r[j] - centroid[j]))
                .collect();
            let f_e = f(&x_e);
            nfev += 1;

            if f_e < f_r {
                simplex[worst_idx] = x_e;
                f_vals[worst_idx] = f_e;
            } else {
                simplex[worst_idx] = x_r;
                f_vals[worst_idx] = f_r;
            }
        } else if f_r < f_vals[second_worst_idx] {
            // Accept reflection
            simplex[worst_idx] = x_r;
            f_vals[worst_idx] = f_r;
        } else {
            // Contraction
            let (x_c, use_reflection) = if f_r < f_vals[worst_idx] {
                // Outside contraction
                let x_c: Vec<T> = (0..n)
                    .map(|j| centroid[j] + rho * (x_r[j] - centroid[j]))
                    .collect();
                (x_c, true)
            } else {
                // Inside contraction
                let x_c: Vec<T> = (0..n)
                    .map(|j| centroid[j] - rho * (simplex[worst_idx][j] - centroid[j]))
                    .collect();
                (x_c, false)
            };

            let f_c = f(&x_c);
            nfev += 1;

            if (use_reflection && f_c < f_r) || (!use_reflection && f_c < f_vals[worst_idx]) {
                simplex[worst_idx] = x_c;
                f_vals[worst_idx] = f_c;
            } else {
                // Shrink simplex toward best point
                for i in 1..=n {
                    let idx = indices[i];
                    for j in 0..n {
                        simplex[idx][j] =
                            simplex[best_idx][j] + sigma * (simplex[idx][j] - simplex[best_idx][j]);
                    }
                    f_vals[idx] = f(&simplex[idx]);
                    nfev += 1;
                }
            }
        }
    }

    // Find best point
    let best_idx = f_vals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .expect("f_vals should not be empty at end of Nelder-Mead iteration");

    Ok(OptimizeResult {
        x: simplex[best_idx].clone(),
        fun: f_vals[best_idx],
        grad: vec![T::zero(); n],
        nit: cfg.max_iter,
        nfev,
        njev: 0,
        success: false,
        message: "Maximum iterations reached".to_string(),
    })
}
