//! Parameter optimization and model selection for exponential smoothing.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::ArrayView1;

use super::core::ExponentialSmoothing;
use super::helpers::build_and_fit;
use super::types::{
    ExponentialSmoothingResult, InformationCriteria, OptimizationConfig, SeasonalComponent,
    TrendComponent,
};

/// Optimize smoothing parameters by minimizing the sum of squared one-step-ahead
/// prediction errors (SSE) using adaptive grid search.
///
/// This function performs a coarse grid search followed by refinement around the
/// best point found. It supports all model types (SES, Holt, Holt-Winters,
/// damped variants).
///
/// # Arguments
///
/// * `data` - Training time series
/// * `trend` - Trend component type
/// * `seasonal` - Seasonal component type
/// * `period` - Seasonal period (required if seasonal != None)
/// * `config` - Optimization configuration (grid resolution, bounds, etc.)
///
/// # Returns
///
/// A tuple of (best_model, best_result) with optimized parameters.
pub fn optimize_parameters(
    data: &ArrayView1<f64>,
    trend: TrendComponent,
    seasonal: SeasonalComponent,
    period: Option<usize>,
    config: &OptimizationConfig,
) -> Result<(ExponentialSmoothing, ExponentialSmoothingResult)> {
    let n_grid = config.grid_resolution;
    let lo = config.param_min;
    let hi = config.param_max;

    let mut best_sse = f64::INFINITY;
    let mut best_params: (f64, Option<f64>, Option<f64>, Option<f64>) = (0.5, None, None, None);

    let has_trend = trend != TrendComponent::None;
    let has_season = seasonal != SeasonalComponent::None;
    let is_damped = trend == TrendComponent::Damped;

    // Generate parameter grid
    let alpha_grid: Vec<f64> = (0..n_grid)
        .map(|i| lo + (hi - lo) * i as f64 / (n_grid - 1).max(1) as f64)
        .collect();

    let beta_grid: Vec<f64> = if has_trend {
        (0..n_grid)
            .map(|i| lo + (hi - lo) * i as f64 / (n_grid - 1).max(1) as f64)
            .collect()
    } else {
        vec![0.0]
    };

    let gamma_grid: Vec<f64> = if has_season {
        (0..n_grid)
            .map(|i| lo + (hi - lo) * i as f64 / (n_grid - 1).max(1) as f64)
            .collect()
    } else {
        vec![0.0]
    };

    let phi_grid: Vec<f64> = if is_damped {
        // Damping parameter typically 0.8-0.98
        let phi_lo = 0.80_f64.max(lo);
        let phi_hi = 0.98_f64.min(hi);
        let n_phi = (n_grid / 2).max(5);
        (0..n_phi)
            .map(|i| phi_lo + (phi_hi - phi_lo) * i as f64 / (n_phi - 1).max(1) as f64)
            .collect()
    } else {
        vec![1.0]
    };

    // Coarse grid search
    for &a in &alpha_grid {
        for &b in &beta_grid {
            for &g in &gamma_grid {
                for &p in &phi_grid {
                    let model_result = build_and_fit(
                        a,
                        if has_trend { Some(b) } else { None },
                        if has_season { Some(g) } else { None },
                        if is_damped { Some(p) } else { None },
                        trend,
                        seasonal,
                        period,
                        data,
                    );
                    if let Ok(res) = model_result {
                        if res.sse < best_sse && res.sse.is_finite() {
                            best_sse = res.sse;
                            best_params = (
                                a,
                                if has_trend { Some(b) } else { None },
                                if has_season { Some(g) } else { None },
                                if is_damped { Some(p) } else { None },
                            );
                        }
                    }
                }
            }
        }
    }

    // Refinement iterations around the best point
    let mut current_best = best_params;
    let mut current_sse = best_sse;

    for iter in 0..config.refinement_iterations {
        let scale = 1.0 / ((iter + 1) as f64 * n_grid as f64);
        let half_range = (hi - lo) * scale;
        let n_refine = 11_usize;

        let refine_grid = |center: f64| -> Vec<f64> {
            let r_lo = (center - half_range).max(lo);
            let r_hi = (center + half_range).min(hi);
            (0..n_refine)
                .map(|i| r_lo + (r_hi - r_lo) * i as f64 / (n_refine - 1).max(1) as f64)
                .collect()
        };

        let a_refine = refine_grid(current_best.0);
        let b_refine = current_best.1.map_or_else(|| vec![0.0], &refine_grid);
        let g_refine = current_best.2.map_or_else(|| vec![0.0], &refine_grid);
        let p_refine = if is_damped {
            let center = current_best.3.unwrap_or(0.9);
            let pr_lo = (center - half_range).max(0.80);
            let pr_hi = (center + half_range).min(0.98);
            (0..n_refine)
                .map(|i| pr_lo + (pr_hi - pr_lo) * i as f64 / (n_refine - 1).max(1) as f64)
                .collect()
        } else {
            vec![1.0]
        };

        for &a in &a_refine {
            for &b in &b_refine {
                for &g in &g_refine {
                    for &p in &p_refine {
                        let model_result = build_and_fit(
                            a,
                            if has_trend { Some(b) } else { None },
                            if has_season { Some(g) } else { None },
                            if is_damped { Some(p) } else { None },
                            trend,
                            seasonal,
                            period,
                            data,
                        );
                        if let Ok(res) = model_result {
                            if res.sse < current_sse && res.sse.is_finite() {
                                current_sse = res.sse;
                                current_best = (
                                    a,
                                    if has_trend { Some(b) } else { None },
                                    if has_season { Some(g) } else { None },
                                    if is_damped { Some(p) } else { None },
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // Build final model with best parameters
    let best_model = ExponentialSmoothing::custom(
        current_best.0,
        current_best.1,
        current_best.2,
        current_best.3,
        trend,
        seasonal,
        period,
    )?;
    let best_result = best_model.fit(data)?;

    Ok((best_model, best_result))
}

/// Compare multiple model specifications and select the best one by AICc.
///
/// This is useful for automated model selection, trying different
/// combinations of trend and seasonal components.
///
/// # Arguments
///
/// * `data` - Training time series
/// * `period` - Seasonal period (used for seasonal model candidates)
/// * `config` - Optimization configuration
///
/// # Returns
///
/// A tuple of (best_model, best_result, best_criteria) for the model
/// with the lowest AICc.
pub fn select_best_model(
    data: &ArrayView1<f64>,
    period: Option<usize>,
    config: &OptimizationConfig,
) -> Result<(
    ExponentialSmoothing,
    ExponentialSmoothingResult,
    InformationCriteria,
)> {
    let mut candidates: Vec<(TrendComponent, SeasonalComponent)> = vec![
        (TrendComponent::None, SeasonalComponent::None),
        (TrendComponent::Additive, SeasonalComponent::None),
        (TrendComponent::Damped, SeasonalComponent::None),
    ];

    // Add seasonal candidates only if period is specified and data is long enough
    if let Some(p) = period {
        if data.len() >= 2 * p {
            candidates.push((TrendComponent::None, SeasonalComponent::Additive));
            candidates.push((TrendComponent::Additive, SeasonalComponent::Additive));
            candidates.push((TrendComponent::Damped, SeasonalComponent::Additive));

            // Only try multiplicative if all values are positive
            let all_positive = data.iter().all(|&x| x > 0.0);
            if all_positive {
                candidates.push((TrendComponent::None, SeasonalComponent::Multiplicative));
                candidates.push((TrendComponent::Additive, SeasonalComponent::Multiplicative));
                candidates.push((TrendComponent::Damped, SeasonalComponent::Multiplicative));
            }
        }
    }

    let mut best_aicc = f64::INFINITY;
    let mut best_model: Option<ExponentialSmoothing> = None;
    let mut best_result: Option<ExponentialSmoothingResult> = None;
    let mut best_criteria: Option<InformationCriteria> = None;

    for (trend, seasonal) in candidates {
        let p = if seasonal != SeasonalComponent::None {
            period
        } else {
            None
        };

        match optimize_parameters(data, trend, seasonal, p, config) {
            Ok((model, result)) => {
                if let Ok(ic) = model.information_criteria(&result) {
                    if ic.aicc < best_aicc && ic.aicc.is_finite() {
                        best_aicc = ic.aicc;
                        best_model = Some(model);
                        best_result = Some(result);
                        best_criteria = Some(ic);
                    }
                }
            }
            Err(_) => continue, // Skip models that fail to fit
        }
    }

    match (best_model, best_result, best_criteria) {
        (Some(m), Some(r), Some(c)) => Ok((m, r, c)),
        _ => Err(NumRs2Error::ComputationError(
            "No valid model could be fitted to the data".to_string(),
        )),
    }
}
