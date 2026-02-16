//! Change Point Detection for Time Series
//!
//! Implements various algorithms for detecting structural breaks and change points
//! in time series data, including changes in mean, variance, and distribution.
//!
//! ## Algorithms Implemented
//!
//! - **CUSUM** (Cumulative Sum): Forward/backward CUSUM with control limits
//! - **PELT** (Pruned Exact Linear Time): Optimal change point detection with BIC/AIC/MBIC penalties
//! - **Binary Segmentation**: Fast approximate change point detection via recursive splitting
//! - **Bayesian Change Point Detection**: Online Bayesian (Adams & MacKay) with run length probabilities
//!
//! ## Utility Functions
//!
//! - Cost functions: L2, L1, normal log-likelihood
//! - Segment statistics: mean, variance per segment
//! - Multiple change point summary
//!
//! ## References
//!
//! - Killick, R., Fearnhead, P., & Eckley, I. A. (2012). "Optimal detection of changepoints
//!   with a linear computational cost." *JASA*, 107(500), 1590-1598.
//! - Adams, R. P., & MacKay, D. J. C. (2007). "Bayesian online changepoint detection."
//!   *arXiv preprint arXiv:0710.3742*.
//! - Page, E. S. (1954). "Continuous inspection schemes." *Biometrika*, 41(1/2), 100-115.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};

// =============================================================================
// Cost Functions
// =============================================================================

/// Cost function type for change point detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostFunction {
    /// L2 (least squares) cost: sum of squared deviations from segment mean
    L2,
    /// L1 (least absolute deviations) cost: sum of absolute deviations from segment median
    L1,
    /// Normal log-likelihood cost: negative log-likelihood under Gaussian assumption
    NormalLogLikelihood,
}

/// Compute cost for a segment under a given cost function.
///
/// # Arguments
/// * `segment` - The data segment
/// * `cost_fn` - Which cost function to use
///
/// # Returns
/// The cost value for the segment
pub fn segment_cost(segment: &ArrayView1<f64>, cost_fn: CostFunction) -> Result<f64> {
    let n = segment.len();
    if n == 0 {
        return Ok(0.0);
    }

    match cost_fn {
        CostFunction::L2 => {
            let mean = segment.iter().sum::<f64>() / n as f64;
            let cost = segment.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
            Ok(cost)
        }
        CostFunction::L1 => {
            let mut sorted: Vec<f64> = segment.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if n.is_multiple_of(2) {
                (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            } else {
                sorted[n / 2]
            };
            let cost = segment.iter().map(|&x| (x - median).abs()).sum::<f64>();
            Ok(cost)
        }
        CostFunction::NormalLogLikelihood => {
            let nf = n as f64;
            let mean = segment.iter().sum::<f64>() / nf;
            let variance = segment.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nf;
            if variance <= 1e-15 {
                // Near-zero variance: assign a large but finite cost
                return Ok(nf * 50.0);
            }
            // Negative log-likelihood: (n/2)*ln(2*pi*sigma^2) + n/2
            Ok(0.5 * nf * (2.0 * std::f64::consts::PI * variance).ln() + 0.5 * nf)
        }
    }
}

// =============================================================================
// Penalty Functions
// =============================================================================

/// Penalty type for model selection in change point detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyType {
    /// Bayesian Information Criterion: penalty = log(n)
    BIC,
    /// Akaike Information Criterion: penalty = 2
    AIC,
    /// Modified BIC: penalty = 3 * log(n)
    MBIC,
    /// Manual penalty value (stored externally)
    Manual,
}

/// Compute penalty value for a given type.
///
/// # Arguments
/// * `penalty_type` - The penalty type
/// * `n` - Number of data points
/// * `manual_value` - Manual penalty value (used only for `PenaltyType::Manual`)
pub fn compute_penalty(penalty_type: PenaltyType, n: usize, manual_value: f64) -> f64 {
    let nf = n as f64;
    match penalty_type {
        PenaltyType::BIC => nf.ln(),
        PenaltyType::AIC => 2.0,
        PenaltyType::MBIC => 3.0 * nf.ln(),
        PenaltyType::Manual => manual_value,
    }
}

// =============================================================================
// Segment Statistics
// =============================================================================

/// Statistics for a single segment of a time series.
#[derive(Debug, Clone)]
pub struct SegmentStats {
    /// Start index of the segment (inclusive)
    pub start: usize,
    /// End index of the segment (exclusive)
    pub end: usize,
    /// Segment mean
    pub mean: f64,
    /// Segment variance
    pub variance: f64,
    /// Number of observations in the segment
    pub count: usize,
}

/// Compute segment statistics given data and change point locations.
///
/// # Arguments
/// * `data` - The full time series
/// * `change_points` - Sorted change point indices
///
/// # Returns
/// A vector of `SegmentStats`, one per segment
pub fn compute_segment_statistics(
    data: &ArrayView1<f64>,
    change_points: &[usize],
) -> Result<Vec<SegmentStats>> {
    let n = data.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut boundaries = vec![0];
    for &cp in change_points {
        if cp > 0 && cp < n {
            boundaries.push(cp);
        }
    }
    boundaries.push(n);
    boundaries.sort_unstable();
    boundaries.dedup();

    let mut stats = Vec::with_capacity(boundaries.len() - 1);
    for w in boundaries.windows(2) {
        let start = w[0];
        let end = w[1];
        let count = end - start;
        if count == 0 {
            continue;
        }
        let segment = data.slice(s![start..end]);
        let mean = segment.iter().sum::<f64>() / count as f64;
        let variance = if count > 1 {
            segment.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / count as f64
        } else {
            0.0
        };
        stats.push(SegmentStats {
            start,
            end,
            mean,
            variance,
            count,
        });
    }

    Ok(stats)
}

// =============================================================================
// Change Point Result
// =============================================================================

/// Result of change point detection.
#[derive(Debug, Clone)]
pub struct ChangePointResult {
    /// Detected change point locations (indices)
    pub locations: Vec<usize>,
    /// Cost or statistic at each change point
    pub costs: Vec<f64>,
    /// Total optimal cost
    pub total_cost: f64,
}

/// Summary of a multiple change point analysis.
#[derive(Debug, Clone)]
pub struct ChangePointSummary {
    /// Number of change points detected
    pub n_changepoints: usize,
    /// Change point locations
    pub locations: Vec<usize>,
    /// Per-segment statistics
    pub segments: Vec<SegmentStats>,
    /// Total cost
    pub total_cost: f64,
}

/// Produce a summary of a change point detection result.
///
/// # Arguments
/// * `data` - The original time series
/// * `result` - The detection result
pub fn summarize_changepoints(
    data: &ArrayView1<f64>,
    result: &ChangePointResult,
) -> Result<ChangePointSummary> {
    let segments = compute_segment_statistics(data, &result.locations)?;
    Ok(ChangePointSummary {
        n_changepoints: result.locations.len(),
        locations: result.locations.clone(),
        segments,
        total_cost: result.total_cost,
    })
}

// =============================================================================
// CUSUM (Cumulative Sum) Detection
// =============================================================================

/// CUSUM (Cumulative Sum) control chart for change detection.
///
/// Implements both forward and backward CUSUM for detecting mean shifts.
/// The two-sided CUSUM tracks both upward and downward shifts simultaneously:
///
/// - S_high(t) = max(0, S_high(t-1) + (x_t - mu) - k)
/// - S_low(t) = max(0, S_low(t-1) - (x_t - mu) - k)
///
/// A change point is detected when either S_high or S_low exceeds the threshold h.
///
/// # References
///
/// Page, E. S. (1954). "Continuous inspection schemes." *Biometrika*, 41(1/2), 100-115.
#[derive(Debug, Clone)]
pub struct Cusum {
    /// Target mean (reference value)
    pub target: f64,
    /// Threshold for detection (h parameter)
    pub threshold: f64,
    /// Drift parameter (k parameter, typically 0.5 * shift_size)
    pub drift: f64,
}

impl Cusum {
    /// Create a new CUSUM detector.
    ///
    /// # Arguments
    /// * `target` - Target mean (reference value)
    /// * `threshold` - Detection threshold (h), typically 4-5 standard deviations
    /// * `drift` - Drift parameter (k), typically 0.5 * expected shift size
    pub fn new(target: f64, threshold: f64, drift: f64) -> Self {
        Self {
            target,
            threshold,
            drift,
        }
    }

    /// Detect change points using two-sided CUSUM.
    ///
    /// Returns all time indices where the CUSUM statistic exceeds the threshold.
    pub fn detect(&self, data: &ArrayView1<f64>) -> Result<ChangePointResult> {
        if data.is_empty() {
            return Ok(ChangePointResult {
                locations: Vec::new(),
                costs: Vec::new(),
                total_cost: 0.0,
            });
        }

        let mut locations = Vec::new();
        let mut costs = Vec::new();
        let mut s_high = 0.0_f64;
        let mut s_low = 0.0_f64;

        for (i, &x) in data.iter().enumerate() {
            s_high = (s_high + (x - self.target) - self.drift).max(0.0);
            s_low = (s_low - (x - self.target) - self.drift).max(0.0);

            if s_high > self.threshold {
                locations.push(i);
                costs.push(s_high);
                s_high = 0.0;
            } else if s_low > self.threshold {
                locations.push(i);
                costs.push(s_low);
                s_low = 0.0;
            }
        }

        let total_cost = costs.iter().sum();
        Ok(ChangePointResult {
            locations,
            costs,
            total_cost,
        })
    }

    /// Compute forward CUSUM statistics for the entire series.
    ///
    /// Returns the cumulative sum statistics (high and low) at each time step.
    pub fn forward_cusum(&self, data: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let n = data.len();
        if n == 0 {
            return Ok((Array1::zeros(0), Array1::zeros(0)));
        }

        let mut s_high = Array1::zeros(n);
        let mut s_low = Array1::zeros(n);

        s_high[0] = ((data[0] - self.target) - self.drift).max(0.0);
        s_low[0] = (-(data[0] - self.target) - self.drift).max(0.0);

        for i in 1..n {
            s_high[i] = (s_high[i - 1] + (data[i] - self.target) - self.drift).max(0.0);
            s_low[i] = (s_low[i - 1] - (data[i] - self.target) - self.drift).max(0.0);
        }

        Ok((s_high, s_low))
    }

    /// Compute backward CUSUM statistics (running from end to start).
    ///
    /// This is useful for confirming change points detected by forward CUSUM.
    pub fn backward_cusum(&self, data: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let n = data.len();
        if n == 0 {
            return Ok((Array1::zeros(0), Array1::zeros(0)));
        }

        let mut s_high = Array1::zeros(n);
        let mut s_low = Array1::zeros(n);

        let last = n - 1;
        s_high[last] = ((data[last] - self.target) - self.drift).max(0.0);
        s_low[last] = (-(data[last] - self.target) - self.drift).max(0.0);

        for i in (0..last).rev() {
            s_high[i] = (s_high[i + 1] + (data[i] - self.target) - self.drift).max(0.0);
            s_low[i] = (s_low[i + 1] - (data[i] - self.target) - self.drift).max(0.0);
        }

        Ok((s_high, s_low))
    }

    /// Compute control limits for the CUSUM chart.
    ///
    /// Returns (upper_limit, lower_limit) arrays. A value crossing the control
    /// limit indicates an out-of-control condition.
    pub fn control_limits(&self, n: usize) -> (Array1<f64>, Array1<f64>) {
        let upper = Array1::from_elem(n, self.threshold);
        let lower = Array1::from_elem(n, self.threshold);
        (upper, lower)
    }
}

// =============================================================================
// PELT (Pruned Exact Linear Time)
// =============================================================================

/// Change type for PELT detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// Detect changes in mean only
    Mean,
    /// Detect changes in variance only
    Variance,
    /// Detect changes in both mean and variance
    MeanAndVariance,
}

/// PELT (Pruned Exact Linear Time) algorithm for optimal change point detection.
///
/// Efficiently finds the optimal set of change points by dynamic programming
/// with a pruning step that achieves O(n) average time complexity.
///
/// Cost function: sum[C(y_{tau_i+1:tau_{i+1}})] + beta * K
/// where C is the segment cost, beta is penalty, and K is number of change points.
///
/// # References
///
/// Killick, R., Fearnhead, P., & Eckley, I. A. (2012). "Optimal detection of changepoints
/// with a linear computational cost." *JASA*, 107(500), 1590-1598.
#[derive(Debug, Clone)]
pub struct Pelt {
    /// Penalty for each change point
    pub penalty: f64,
    /// Minimum segment length between change points
    pub min_size: usize,
    /// Cost function to use
    pub cost_fn: CostFunction,
    /// Penalty type (for automatic penalty computation)
    pub penalty_type: PenaltyType,
}

impl Pelt {
    /// Create a new PELT detector with manual penalty.
    ///
    /// # Arguments
    /// * `penalty` - Penalty for each change point (typical: log(n) to 2*log(n))
    /// * `min_size` - Minimum segment length (default: 2)
    pub fn new(penalty: f64, min_size: usize) -> Self {
        Self {
            penalty,
            min_size: min_size.max(1),
            cost_fn: CostFunction::NormalLogLikelihood,
            penalty_type: PenaltyType::Manual,
        }
    }

    /// Create PELT with a specific penalty type and cost function.
    ///
    /// The penalty value will be computed automatically from the data length.
    pub fn with_options(penalty_type: PenaltyType, cost_fn: CostFunction, min_size: usize) -> Self {
        Self {
            penalty: 0.0, // will be computed from data
            min_size: min_size.max(1),
            cost_fn,
            penalty_type,
        }
    }

    /// Detect change points based on the configured change type.
    pub fn detect(
        &self,
        data: &ArrayView1<f64>,
        change_type: ChangeType,
    ) -> Result<ChangePointResult> {
        match change_type {
            ChangeType::Mean => self.detect_mean_change(data),
            ChangeType::Variance => self.detect_variance_change(data),
            ChangeType::MeanAndVariance => self.detect_mean_variance_change(data),
        }
    }

    /// Detect change points in mean.
    pub fn detect_mean_change(&self, data: &ArrayView1<f64>) -> Result<ChangePointResult> {
        self.pelt_core(data, CostFunction::L2)
    }

    /// Detect change points in variance.
    pub fn detect_variance_change(&self, data: &ArrayView1<f64>) -> Result<ChangePointResult> {
        self.pelt_core(data, CostFunction::NormalLogLikelihood)
    }

    /// Detect change points in both mean and variance.
    pub fn detect_mean_variance_change(&self, data: &ArrayView1<f64>) -> Result<ChangePointResult> {
        self.pelt_core(data, CostFunction::NormalLogLikelihood)
    }

    /// Core PELT dynamic programming algorithm.
    fn pelt_core(
        &self,
        data: &ArrayView1<f64>,
        cost_fn: CostFunction,
    ) -> Result<ChangePointResult> {
        let n = data.len();
        if n < 2 * self.min_size {
            return Err(NumRs2Error::ValueError(
                "Insufficient data for change point detection".to_string(),
            ));
        }

        // Resolve penalty
        let penalty = if self.penalty_type == PenaltyType::Manual {
            self.penalty
        } else {
            compute_penalty(self.penalty_type, n, self.penalty)
        };

        // Dynamic programming arrays
        // f[t] = optimal cost for data[0..t]
        let mut f = vec![f64::INFINITY; n + 1];
        let mut last_cp = vec![0_usize; n + 1]; // last change point for backtracking
        f[0] = -penalty;

        // Candidate set for pruning
        let mut candidates: Vec<usize> = vec![0];

        for t in self.min_size..=n {
            let mut best_cost = f64::INFINITY;
            let mut best_s = 0_usize;

            for &s_val in &candidates {
                if t - s_val >= self.min_size {
                    let seg = data.slice(s![s_val..t]);
                    let c = segment_cost(&seg, cost_fn)?;
                    let total = f[s_val] + c + penalty;
                    if total < best_cost {
                        best_cost = total;
                        best_s = s_val;
                    }
                }
            }

            if best_cost.is_infinite() {
                // No valid candidate yet; keep going
                candidates.push(t);
                continue;
            }

            f[t] = best_cost;
            last_cp[t] = best_s;

            // Pruning step: retain only candidates that can still be optimal
            candidates.retain(|&s_val| {
                if t - s_val < self.min_size {
                    return true;
                }
                if let Ok(c) = segment_cost(&data.slice(s![s_val..t]), cost_fn) {
                    f[s_val] + c + penalty <= best_cost
                } else {
                    false
                }
            });

            candidates.push(t);
        }

        // Backtrack to find change points
        let mut locations = Vec::new();
        let mut idx = n;
        while idx > 0 {
            let prev = last_cp[idx];
            if prev > 0 {
                locations.push(prev);
            }
            if prev >= idx {
                break;
            }
            idx = prev;
        }
        locations.sort_unstable();
        locations.dedup();

        // Compute per-segment costs
        let costs = self.compute_costs_for_segments(data, &locations, cost_fn)?;
        let total_cost = if f[n].is_finite() { f[n] } else { 0.0 };

        Ok(ChangePointResult {
            locations,
            costs,
            total_cost,
        })
    }

    /// Compute costs for each segment defined by change points.
    fn compute_costs_for_segments(
        &self,
        data: &ArrayView1<f64>,
        change_points: &[usize],
        cost_fn: CostFunction,
    ) -> Result<Vec<f64>> {
        let n = data.len();
        let mut costs = Vec::new();
        let mut start = 0;

        for &cp in change_points {
            if cp > start && cp <= n {
                let seg = data.slice(s![start..cp]);
                costs.push(segment_cost(&seg, cost_fn)?);
                start = cp;
            }
        }

        if start < n {
            let seg = data.slice(s![start..n]);
            costs.push(segment_cost(&seg, cost_fn)?);
        }

        Ok(costs)
    }
}

// =============================================================================
// Binary Segmentation
// =============================================================================

/// Binary Segmentation for change point detection.
///
/// Greedy algorithm that recursively splits segments at points of maximum change
/// statistic. Time complexity: O(n log n) average, O(n^2) worst case.
///
/// # Algorithm
///
/// 1. Compute test statistic for all candidate split points across the full series.
/// 2. If the maximum statistic exceeds the threshold, accept the split point.
/// 3. Recursively apply to each resulting sub-segment.
/// 4. Stop when no segment can be split or max_changepoints is reached.
#[derive(Debug, Clone)]
pub struct BinarySegmentation {
    /// Maximum number of change points to detect
    pub max_changepoints: usize,
    /// Threshold for accepting a change point
    pub threshold: f64,
    /// Minimum segment size
    pub min_size: usize,
    /// Cost function
    pub cost_fn: CostFunction,
}

impl BinarySegmentation {
    /// Create a new binary segmentation detector.
    pub fn new(max_changepoints: usize, threshold: f64, min_size: usize) -> Self {
        Self {
            max_changepoints,
            threshold,
            min_size: min_size.max(1),
            cost_fn: CostFunction::L2,
        }
    }

    /// Create with a specific cost function.
    pub fn with_cost(
        max_changepoints: usize,
        threshold: f64,
        min_size: usize,
        cost_fn: CostFunction,
    ) -> Self {
        Self {
            max_changepoints,
            threshold,
            min_size: min_size.max(1),
            cost_fn,
        }
    }

    /// Detect change points in the data.
    pub fn detect(&self, data: &ArrayView1<f64>) -> Result<ChangePointResult> {
        let n = data.len();
        if n < 2 * self.min_size {
            return Ok(ChangePointResult {
                locations: Vec::new(),
                costs: Vec::new(),
                total_cost: 0.0,
            });
        }

        let mut change_points = Vec::new();
        let mut segments: Vec<(usize, usize)> = vec![(0, n)];
        let mut total_cost = 0.0;

        for _ in 0..self.max_changepoints {
            let mut best_seg_idx = None;
            let mut best_cp = 0;
            let mut best_gain = 0.0_f64;

            for (idx, &(start, end)) in segments.iter().enumerate() {
                if end - start < 2 * self.min_size {
                    continue;
                }

                if let Ok((cp, gain)) = self.find_best_split(data, start, end) {
                    if gain > best_gain {
                        best_gain = gain;
                        best_cp = cp;
                        best_seg_idx = Some(idx);
                    }
                }
            }

            if best_gain < self.threshold {
                break;
            }

            if let Some(idx) = best_seg_idx {
                let (start, end) = segments[idx];
                segments.remove(idx);
                segments.push((start, best_cp));
                segments.push((best_cp, end));
                change_points.push(best_cp);
                total_cost += best_gain;
            } else {
                break;
            }
        }

        change_points.sort_unstable();

        let costs = if change_points.is_empty() {
            Vec::new()
        } else {
            vec![total_cost / change_points.len() as f64; change_points.len()]
        };

        Ok(ChangePointResult {
            locations: change_points,
            costs,
            total_cost,
        })
    }

    /// Find the best split point within a segment [start, end).
    ///
    /// The gain is computed as: cost(full_segment) - cost(left) - cost(right)
    fn find_best_split(
        &self,
        data: &ArrayView1<f64>,
        start: usize,
        end: usize,
    ) -> Result<(usize, f64)> {
        if end - start < 2 * self.min_size {
            return Err(NumRs2Error::ValueError("Segment too small".to_string()));
        }

        let full_seg = data.slice(s![start..end]);
        let full_cost = segment_cost(&full_seg, self.cost_fn)?;

        let mut best_cp = start + self.min_size;
        let mut best_gain = f64::NEG_INFINITY;

        for t in (start + self.min_size)..(end - self.min_size + 1) {
            let left = data.slice(s![start..t]);
            let right = data.slice(s![t..end]);

            let left_cost = segment_cost(&left, self.cost_fn)?;
            let right_cost = segment_cost(&right, self.cost_fn)?;

            let gain = full_cost - left_cost - right_cost;
            if gain > best_gain {
                best_gain = gain;
                best_cp = t;
            }
        }

        Ok((best_cp, best_gain))
    }
}

// =============================================================================
// Bayesian Online Change Point Detection (Adams & MacKay 2007)
// =============================================================================

/// Bayesian Online Change Point Detection.
///
/// Implements the Adams & MacKay (2007) algorithm that maintains a distribution
/// over "run lengths" (time since last change point). At each time step:
///
/// 1. Observe new data point x_t
/// 2. Update run length probabilities using predictive likelihood
/// 3. Apply hazard function to compute growth and change point probabilities
///
/// The hazard function H(tau) gives the probability of a change point given the
/// current run length tau. A constant hazard 1/lambda yields a geometric prior
/// on run lengths.
///
/// # References
///
/// Adams, R. P., & MacKay, D. J. C. (2007). "Bayesian online changepoint detection."
/// *arXiv preprint arXiv:0710.3742*.
#[derive(Debug, Clone)]
pub struct BayesianChangePoint {
    /// Hazard rate (1/lambda): probability of change at each time step
    pub hazard_rate: f64,
    /// Prior mean for the normal model
    pub prior_mean: f64,
    /// Prior variance for the normal model
    pub prior_var: f64,
    /// Observation noise variance
    pub obs_var: f64,
    /// Threshold on change point posterior probability
    pub threshold: f64,
}

impl BayesianChangePoint {
    /// Create a new Bayesian change point detector.
    ///
    /// # Arguments
    /// * `hazard_rate` - Constant hazard rate (1/lambda). Higher values mean change
    ///   points are more frequent a priori.
    /// * `threshold` - Posterior probability threshold for declaring a change point.
    pub fn new(hazard_rate: f64, threshold: f64) -> Self {
        Self {
            hazard_rate: hazard_rate.clamp(1e-10, 1.0 - 1e-10),
            prior_mean: 0.0,
            prior_var: 1.0,
            obs_var: 1.0,
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Create with custom prior parameters.
    pub fn with_prior(
        hazard_rate: f64,
        threshold: f64,
        prior_mean: f64,
        prior_var: f64,
        obs_var: f64,
    ) -> Self {
        Self {
            hazard_rate: hazard_rate.clamp(1e-10, 1.0 - 1e-10),
            prior_mean,
            prior_var: prior_var.max(1e-10),
            obs_var: obs_var.max(1e-10),
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Detect change points using Bayesian online detection.
    ///
    /// Uses the MAP (Maximum A Posteriori) run length to detect changepoints.
    /// With a constant hazard function, P(r_t = 0 | x_{1:t}) = h always,
    /// so the raw run-length-zero probability cannot distinguish changepoints.
    /// Instead, we detect times when the MAP run length drops significantly,
    /// indicating the run length distribution has shifted to short run lengths
    /// after a structural break in the data.
    ///
    /// The `threshold` parameter controls the minimum posterior probability
    /// of the new MAP run length required to declare a changepoint, filtering
    /// out low-confidence detections.
    pub fn detect(&self, data: &ArrayView1<f64>) -> Result<ChangePointResult> {
        let n = data.len();
        if n < 2 {
            return Ok(ChangePointResult {
                locations: Vec::new(),
                costs: Vec::new(),
                total_cost: 0.0,
            });
        }

        let (rlp, _cp_probs) = self.compute_run_length_distribution(data)?;

        // Compute the MAP run length and its posterior probability at each time step
        let mut map_rls = vec![0_usize; n + 1];
        let mut map_probs = vec![0.0_f64; n + 1];
        for t in 0..=n {
            let mut best_r = 0_usize;
            let mut best_p = 0.0_f64;
            let max_r = t.min(n);
            for r in 0..=max_r {
                if rlp[[t, r]] > best_p {
                    best_p = rlp[[t, r]];
                    best_r = r;
                }
            }
            map_rls[t] = best_r;
            map_probs[t] = best_p;
        }

        let mut locations = Vec::new();
        let mut costs = Vec::new();

        // Detect changepoints: the MAP run length should grow by 1 each step
        // under continuity. A significant drop indicates a changepoint.
        for t in 2..=n {
            let prev_rl = map_rls[t - 1];
            let curr_rl = map_rls[t];

            // Under no changepoint, we expect curr_rl ≈ prev_rl + 1.
            // A changepoint is signalled when the MAP run length drops
            // instead of growing (and the previous run was long enough).
            let expected_rl = prev_rl + 1;
            if expected_rl > 3 && curr_rl < expected_rl / 2 && map_probs[t] >= self.threshold {
                // The changepoint occurred at approximately t - curr_rl
                let cp_location = t.saturating_sub(curr_rl);

                // Deduplicate: skip if too close to a previous detection
                let already_found = locations
                    .iter()
                    .any(|&loc: &usize| loc.abs_diff(cp_location) < 5);

                if !already_found && cp_location > 0 && cp_location < n {
                    locations.push(cp_location);
                    costs.push(map_probs[t]);
                }
            }
        }

        let total_cost = costs.iter().sum();
        Ok(ChangePointResult {
            locations,
            costs,
            total_cost,
        })
    }

    /// Compute the full run length distribution over time.
    ///
    /// Returns:
    /// * `run_length_probs` - Matrix where row t contains the probability of each
    ///   run length at time t. Entry \[t\]\[r\] = P(r_t = r | x_{1:t}).
    /// * `cp_probs` - Vector of change point posterior probabilities at each time step.
    pub fn compute_run_length_distribution(
        &self,
        data: &ArrayView1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>)> {
        let n = data.len();
        if n == 0 {
            return Ok((Array2::zeros((0, 0)), Array1::zeros(0)));
        }

        // run_length_probs[t][r] = joint probability P(r_t = r, x_{1:t})
        // We store up to (n+1) possible run lengths at each time step
        let mut rlp = Array2::zeros((n + 1, n + 1));
        rlp[[0, 0]] = 1.0; // Initially run length is 0 with probability 1

        let mut cp_probs = Array1::zeros(n + 1);

        // Sufficient statistics for the conjugate normal model per run length
        // For each possible run length r: accumulate sum and sum-of-squares
        let mut mu = vec![self.prior_mean; n + 1]; // posterior mean
        let mut sigma2 = vec![self.prior_var; n + 1]; // posterior variance

        for t in 0..n {
            let x = data[t];

            // Compute predictive probabilities for each run length
            let mut pred_probs = vec![0.0_f64; t + 1];
            for r in 0..=t {
                // Predictive distribution: N(mu_r, sigma2_r + obs_var)
                let pred_var = sigma2[r] + self.obs_var;
                let diff = x - mu[r];
                let log_pred = -0.5 * (2.0 * std::f64::consts::PI * pred_var).ln()
                    - 0.5 * diff * diff / pred_var;
                pred_probs[r] = log_pred.exp();
            }

            // Growth probabilities (run length grows by 1)
            let h = self.hazard_rate;
            let mut new_rlp = vec![0.0_f64; t + 2];

            // Growth: run length increases
            for r in 0..=t {
                new_rlp[r + 1] += rlp[[t, r]] * pred_probs[r] * (1.0 - h);
            }

            // Change point: run length resets to 0
            let mut cp_sum = 0.0;
            for r in 0..=t {
                cp_sum += rlp[[t, r]] * pred_probs[r] * h;
            }
            new_rlp[0] = cp_sum;

            // Normalize
            let evidence: f64 = new_rlp.iter().sum();
            if evidence > 1e-300 {
                for r in 0..=(t + 1) {
                    rlp[[t + 1, r]] = new_rlp[r] / evidence;
                }
            }

            cp_probs[t + 1] = if evidence > 1e-300 {
                new_rlp[0] / evidence
            } else {
                0.0
            };

            // Update sufficient statistics for each run length
            // Bayesian update: posterior after seeing x with prior (mu_r, sigma2_r)
            let mut new_mu = vec![self.prior_mean; t + 2];
            let mut new_sigma2 = vec![self.prior_var; t + 2];

            for r in 0..=t {
                let kalman_gain = sigma2[r] / (sigma2[r] + self.obs_var);
                new_mu[r + 1] = mu[r] + kalman_gain * (x - mu[r]);
                new_sigma2[r + 1] = sigma2[r] * (1.0 - kalman_gain);
            }
            // Run length 0 resets to prior
            new_mu[0] = self.prior_mean;
            new_sigma2[0] = self.prior_var;

            mu = new_mu;
            sigma2 = new_sigma2;
        }

        // Extract the posterior portion we filled
        let rlp_out = rlp.slice(s![0..=n, 0..=n]).to_owned();
        let cp_out = cp_probs.slice(s![0..=n]).to_owned();

        Ok((rlp_out, cp_out))
    }

    /// Constant hazard function value.
    pub fn hazard(&self, _run_length: usize) -> f64 {
        self.hazard_rate
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    // -------------------------------------------------------------------------
    // Helper: generate step signal with a single mean shift
    // -------------------------------------------------------------------------
    fn step_signal(n1: usize, n2: usize, mean1: f64, mean2: f64) -> Array1<f64> {
        let mut data = vec![mean1; n1];
        data.extend(vec![mean2; n2]);
        Array1::from_vec(data)
    }

    /// Add small deterministic noise to avoid zero-variance segments.
    fn add_noise(data: &mut [f64]) {
        for (i, v) in data.iter_mut().enumerate() {
            *v += 0.01 * ((i as f64 * 1.23456).sin());
        }
    }

    // -------------------------------------------------------------------------
    // Test 1: CUSUM - single mean shift detection
    // -------------------------------------------------------------------------
    #[test]
    fn test_cusum_single_mean_shift() {
        let mut data = vec![0.0; 50];
        data.extend(vec![3.0; 50]);
        add_noise(&mut data);
        let arr = Array1::from_vec(data);

        let cusum = Cusum::new(0.0, 8.0, 0.5);
        let result = cusum
            .detect(&arr.view())
            .expect("CUSUM detect should succeed");

        assert!(
            !result.locations.is_empty(),
            "CUSUM should detect at least one change point"
        );
        // First detection should be near index 50
        let first = result.locations[0];
        assert!(
            (40..=70).contains(&first),
            "First change point {first} should be near 50"
        );
    }

    // -------------------------------------------------------------------------
    // Test 2: CUSUM forward and backward
    // -------------------------------------------------------------------------
    #[test]
    fn test_cusum_forward_backward() {
        let mut data = vec![0.0; 30];
        data.extend(vec![4.0; 30]);
        add_noise(&mut data);
        let arr = Array1::from_vec(data);

        let cusum = Cusum::new(0.0, 5.0, 0.5);
        let (fwd_high, fwd_low) = cusum
            .forward_cusum(&arr.view())
            .expect("forward CUSUM should succeed");
        let (bwd_high, bwd_low) = cusum
            .backward_cusum(&arr.view())
            .expect("backward CUSUM should succeed");

        assert_eq!(fwd_high.len(), 60);
        assert_eq!(bwd_high.len(), 60);

        // Forward high CUSUM should rise after the shift
        assert!(
            fwd_high[59] > fwd_high[0],
            "Forward CUSUM should rise after shift"
        );
        // Backward high CUSUM should be large at the beginning (running backwards from shift)
        assert!(
            bwd_high[0] > bwd_high[59],
            "Backward CUSUM should detect shift from the other direction"
        );
    }

    // -------------------------------------------------------------------------
    // Test 3: CUSUM control limits
    // -------------------------------------------------------------------------
    #[test]
    fn test_cusum_control_limits() {
        let cusum = Cusum::new(0.0, 5.0, 0.5);
        let (upper, lower) = cusum.control_limits(100);
        assert_eq!(upper.len(), 100);
        assert_eq!(lower.len(), 100);
        for i in 0..100 {
            assert!((upper[i] - 5.0).abs() < 1e-10);
            assert!((lower[i] - 5.0).abs() < 1e-10);
        }
    }

    // -------------------------------------------------------------------------
    // Test 4: CUSUM - no change in stationary data
    // -------------------------------------------------------------------------
    #[test]
    fn test_cusum_no_change() {
        let mut data = vec![5.0; 100];
        add_noise(&mut data);
        let arr = Array1::from_vec(data);

        // Target at 5.0, high threshold so no false alarms
        let cusum = Cusum::new(5.0, 50.0, 0.5);
        let result = cusum.detect(&arr.view()).expect("CUSUM should succeed");

        assert!(
            result.locations.is_empty(),
            "CUSUM should not detect changes in stationary data with high threshold"
        );
    }

    // -------------------------------------------------------------------------
    // Test 5: PELT - single mean change
    // -------------------------------------------------------------------------
    #[test]
    fn test_pelt_single_mean_change() {
        let mut raw = vec![0.0; 50];
        raw.extend(vec![5.0; 50]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let pelt = Pelt::new(15.0, 5);
        let result = pelt
            .detect_mean_change(&data.view())
            .expect("PELT mean change should succeed");

        assert!(
            !result.locations.is_empty(),
            "PELT should detect at least one change point"
        );
        // Should be near index 50
        let closest = result
            .locations
            .iter()
            .min_by_key(|&&cp| ((cp as i64) - 50).unsigned_abs())
            .copied();
        if let Some(cp) = closest {
            assert!(
                (cp as i64 - 50).unsigned_abs() < 10,
                "Closest change point {cp} should be near 50"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test 6: PELT - multiple change points
    // -------------------------------------------------------------------------
    #[test]
    fn test_pelt_multiple_changepoints() {
        let mut raw = vec![0.0; 40];
        raw.extend(vec![5.0; 40]);
        raw.extend(vec![10.0; 40]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let pelt = Pelt::new(15.0, 5);
        let result = pelt
            .detect_mean_change(&data.view())
            .expect("PELT should succeed");

        assert!(
            result.locations.len() >= 2,
            "PELT should detect at least 2 change points, got {}",
            result.locations.len()
        );
    }

    // -------------------------------------------------------------------------
    // Test 7: PELT - variance change
    // -------------------------------------------------------------------------
    #[test]
    fn test_pelt_variance_change() {
        // Low variance segment then high variance segment
        let mut raw = Vec::with_capacity(100);
        for i in 0..50 {
            raw.push(5.0 + 0.05 * (i as f64 * 0.73).sin());
        }
        for i in 50..100 {
            raw.push(5.0 + 3.0 * (i as f64 * 0.73).sin());
        }
        let data = Array1::from_vec(raw);

        let pelt = Pelt::new(10.0, 5);
        let result = pelt
            .detect_variance_change(&data.view())
            .expect("PELT variance change should succeed");

        assert!(result.total_cost.is_finite(), "Total cost should be finite");
    }

    // -------------------------------------------------------------------------
    // Test 8: PELT with BIC penalty
    // -------------------------------------------------------------------------
    #[test]
    fn test_pelt_bic_penalty() {
        let mut raw = vec![0.0; 60];
        raw.extend(vec![6.0; 60]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let pelt = Pelt::with_options(PenaltyType::BIC, CostFunction::L2, 5);
        let result = pelt
            .detect(&data.view(), ChangeType::Mean)
            .expect("PELT with BIC should succeed");

        assert!(
            !result.locations.is_empty(),
            "PELT with BIC should detect change"
        );
    }

    // -------------------------------------------------------------------------
    // Test 9: PELT with AIC penalty
    // -------------------------------------------------------------------------
    #[test]
    fn test_pelt_aic_penalty() {
        let mut raw = vec![0.0; 50];
        raw.extend(vec![5.0; 50]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let pelt = Pelt::with_options(PenaltyType::AIC, CostFunction::L2, 5);
        let result = pelt
            .detect(&data.view(), ChangeType::Mean)
            .expect("PELT with AIC should succeed");

        assert!(
            !result.locations.is_empty(),
            "PELT with AIC should detect change"
        );
    }

    // -------------------------------------------------------------------------
    // Test 10: PELT - no change (stationary)
    // -------------------------------------------------------------------------
    #[test]
    fn test_pelt_no_change_stationary() {
        let mut raw = vec![5.0; 100];
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let pelt = Pelt::new(100.0, 10); // Very high penalty
        let result = pelt
            .detect_mean_change(&data.view())
            .expect("PELT should succeed on stationary data");

        assert!(
            result.locations.is_empty(),
            "PELT should not detect changes in stationary data with high penalty, got {:?}",
            result.locations
        );
    }

    // -------------------------------------------------------------------------
    // Test 11: PELT insufficient data
    // -------------------------------------------------------------------------
    #[test]
    fn test_pelt_insufficient_data() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let pelt = Pelt::new(5.0, 5);
        let result = pelt.detect_mean_change(&data.view());
        assert!(result.is_err(), "PELT should fail with insufficient data");
    }

    // -------------------------------------------------------------------------
    // Test 12: Binary Segmentation - basic detection
    // -------------------------------------------------------------------------
    #[test]
    fn test_binseg_basic_detection() {
        let mut raw = vec![0.0; 50];
        raw.extend(vec![5.0; 50]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let bs = BinarySegmentation::new(5, 5.0, 5);
        let result = bs.detect(&data.view()).expect("BinSeg should succeed");

        assert!(!result.locations.is_empty(), "BinSeg should detect change");
    }

    // -------------------------------------------------------------------------
    // Test 13: Binary Segmentation - multiple change points
    // -------------------------------------------------------------------------
    #[test]
    fn test_binseg_multiple_changepoints() {
        let mut raw = vec![0.0; 40];
        raw.extend(vec![4.0; 40]);
        raw.extend(vec![8.0; 40]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let bs = BinarySegmentation::new(10, 5.0, 5);
        let result = bs.detect(&data.view()).expect("BinSeg should succeed");

        assert!(
            result.locations.len() >= 2,
            "BinSeg should detect at least 2 change points, got {}",
            result.locations.len()
        );
    }

    // -------------------------------------------------------------------------
    // Test 14: PELT vs BinSeg comparison
    // -------------------------------------------------------------------------
    #[test]
    fn test_pelt_vs_binseg_comparison() {
        let mut raw = vec![0.0; 50];
        raw.extend(vec![5.0; 50]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let pelt = Pelt::new(10.0, 5);
        let pelt_result = pelt
            .detect_mean_change(&data.view())
            .expect("PELT should succeed");

        let bs = BinarySegmentation::new(5, 5.0, 5);
        let bs_result = bs.detect(&data.view()).expect("BinSeg should succeed");

        // Both should detect at least one change point
        assert!(
            !pelt_result.locations.is_empty(),
            "PELT should detect change"
        );
        assert!(
            !bs_result.locations.is_empty(),
            "BinSeg should detect change"
        );

        // Both should find a change point near index 50
        let pelt_near_50 = pelt_result
            .locations
            .iter()
            .any(|&cp| (cp as i64 - 50).unsigned_abs() < 15);
        let bs_near_50 = bs_result
            .locations
            .iter()
            .any(|&cp| (cp as i64 - 50).unsigned_abs() < 15);

        assert!(pelt_near_50, "PELT should find change near 50");
        assert!(bs_near_50, "BinSeg should find change near 50");
    }

    // -------------------------------------------------------------------------
    // Test 15: Bayesian online change point detection
    // -------------------------------------------------------------------------
    #[test]
    fn test_bayesian_online_detection() {
        let mut raw = vec![0.0; 50];
        raw.extend(vec![5.0; 50]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let bayes = BayesianChangePoint::with_prior(0.05, 0.3, 0.0, 10.0, 1.0);
        let result = bayes
            .detect(&data.view())
            .expect("Bayesian detection should succeed");

        // Should detect change near index 50
        assert!(
            !result.locations.is_empty(),
            "Bayesian should detect at least one change point"
        );
    }

    // -------------------------------------------------------------------------
    // Test 16: Bayesian run length distribution
    // -------------------------------------------------------------------------
    #[test]
    fn test_bayesian_run_length_distribution() {
        let mut raw = vec![0.0; 30];
        raw.extend(vec![5.0; 30]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let bayes = BayesianChangePoint::with_prior(0.1, 0.3, 0.0, 10.0, 1.0);
        let (rlp, cp_probs) = bayes
            .compute_run_length_distribution(&data.view())
            .expect("Run length computation should succeed");

        assert_eq!(rlp.nrows(), 61); // n+1
        assert_eq!(cp_probs.len(), 61);

        // Run length probabilities should sum to approximately 1 at each time step
        for t in 1..=60 {
            let row_sum: f64 = rlp.row(t).iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 0.1,
                "Row {t} sum = {row_sum} should be near 1.0"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Test 17: Bayesian hazard function
    // -------------------------------------------------------------------------
    #[test]
    fn test_bayesian_hazard_function() {
        let bayes = BayesianChangePoint::new(0.05, 0.5);
        // Constant hazard
        assert!((bayes.hazard(0) - 0.05).abs() < 1e-10);
        assert!((bayes.hazard(100) - 0.05).abs() < 1e-10);
        assert!((bayes.hazard(1000) - 0.05).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // Test 18: Cost functions
    // -------------------------------------------------------------------------
    #[test]
    fn test_cost_functions() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let l2 = segment_cost(&data.view(), CostFunction::L2).expect("L2 cost should succeed");
        assert!(l2 > 0.0, "L2 cost should be positive for non-constant data");

        let l1 = segment_cost(&data.view(), CostFunction::L1).expect("L1 cost should succeed");
        assert!(l1 > 0.0, "L1 cost should be positive for non-constant data");

        let nll = segment_cost(&data.view(), CostFunction::NormalLogLikelihood)
            .expect("NLL cost should succeed");
        assert!(nll.is_finite(), "NLL cost should be finite");
    }

    // -------------------------------------------------------------------------
    // Test 19: Segment statistics
    // -------------------------------------------------------------------------
    #[test]
    fn test_segment_statistics() {
        let data = step_signal(50, 50, 0.0, 10.0);
        let change_points = vec![50];
        let stats = compute_segment_statistics(&data.view(), &change_points)
            .expect("segment stats should succeed");

        assert_eq!(stats.len(), 2);
        assert!((stats[0].mean - 0.0).abs() < 1e-10);
        assert!((stats[1].mean - 10.0).abs() < 1e-10);
        assert_eq!(stats[0].count, 50);
        assert_eq!(stats[1].count, 50);
    }

    // -------------------------------------------------------------------------
    // Test 20: Change point summary
    // -------------------------------------------------------------------------
    #[test]
    fn test_changepoint_summary() {
        let data = step_signal(50, 50, 2.0, 8.0);
        let result = ChangePointResult {
            locations: vec![50],
            costs: vec![10.0],
            total_cost: 10.0,
        };

        let summary =
            summarize_changepoints(&data.view(), &result).expect("summarize should succeed");

        assert_eq!(summary.n_changepoints, 1);
        assert_eq!(summary.segments.len(), 2);
        assert!((summary.segments[0].mean - 2.0).abs() < 1e-10);
        assert!((summary.segments[1].mean - 8.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // Test 21: Penalty functions
    // -------------------------------------------------------------------------
    #[test]
    fn test_penalty_functions() {
        let n = 100;
        let bic = compute_penalty(PenaltyType::BIC, n, 0.0);
        let aic = compute_penalty(PenaltyType::AIC, n, 0.0);
        let mbic = compute_penalty(PenaltyType::MBIC, n, 0.0);
        let manual = compute_penalty(PenaltyType::Manual, n, 42.0);

        assert!((bic - (100.0_f64).ln()).abs() < 1e-10);
        assert!((aic - 2.0).abs() < 1e-10);
        assert!((mbic - 3.0 * (100.0_f64).ln()).abs() < 1e-10);
        assert!((manual - 42.0).abs() < 1e-10);

        // BIC < MBIC for n > e^(2/3)
        assert!(bic < mbic);
    }

    // -------------------------------------------------------------------------
    // Test 22: Edge case - short series
    // -------------------------------------------------------------------------
    #[test]
    fn test_edge_case_short_series() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // PELT with small min_size
        let pelt = Pelt::new(10.0, 2);
        let result = pelt.detect_mean_change(&data.view());
        // Should succeed (5 >= 2*2)
        assert!(result.is_ok());

        // BinSeg with small min_size
        let bs = BinarySegmentation::new(3, 0.5, 2);
        let result = bs.detect(&data.view());
        assert!(result.is_ok());
    }

    // -------------------------------------------------------------------------
    // Test 23: Edge case - change at boundaries
    // -------------------------------------------------------------------------
    #[test]
    fn test_edge_case_change_at_boundary() {
        // Change very early
        let mut early = vec![10.0; 5];
        early.extend(vec![0.0; 95]);
        add_noise(&mut early);
        let data = Array1::from_vec(early);

        let pelt = Pelt::new(10.0, 3);
        let result = pelt
            .detect_mean_change(&data.view())
            .expect("PELT should succeed");
        // Should detect something near the beginning
        assert!(
            !result.locations.is_empty(),
            "Should detect change near beginning"
        );

        // Change very late
        let mut late = vec![0.0; 95];
        late.extend(vec![10.0; 5]);
        add_noise(&mut late);
        let data_late = Array1::from_vec(late);

        let result_late = pelt
            .detect_mean_change(&data_late.view())
            .expect("PELT should succeed on late change");
        assert!(
            !result_late.locations.is_empty(),
            "Should detect change near end"
        );
    }

    // -------------------------------------------------------------------------
    // Test 24: L1 cost function (median-based)
    // -------------------------------------------------------------------------
    #[test]
    fn test_l1_cost_function() {
        let data = Array1::from_vec(vec![1.0, 1.0, 1.0, 10.0, 1.0]);
        let cost = segment_cost(&data.view(), CostFunction::L1).expect("L1 cost should succeed");
        // Median is 1.0, sum of |x - 1| = 0+0+0+9+0 = 9
        assert!(
            (cost - 9.0).abs() < 1e-10,
            "L1 cost should be 9.0, got {cost}"
        );
    }

    // -------------------------------------------------------------------------
    // Test 25: BinarySegmentation with L1 cost
    // -------------------------------------------------------------------------
    #[test]
    fn test_binseg_l1_cost() {
        let mut raw = vec![0.0; 50];
        raw.extend(vec![5.0; 50]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let bs = BinarySegmentation::with_cost(5, 5.0, 5, CostFunction::L1);
        let result = bs.detect(&data.view()).expect("BinSeg L1 should succeed");

        assert!(
            !result.locations.is_empty(),
            "BinSeg with L1 cost should detect change"
        );
    }

    // -------------------------------------------------------------------------
    // Test 26: PELT detect with ChangeType enum
    // -------------------------------------------------------------------------
    #[test]
    fn test_pelt_detect_with_change_type() {
        let mut raw = vec![0.0; 50];
        raw.extend(vec![5.0; 50]);
        add_noise(&mut raw);
        let data = Array1::from_vec(raw);

        let pelt = Pelt::new(10.0, 5);

        let mean_result = pelt
            .detect(&data.view(), ChangeType::Mean)
            .expect("Mean detect should succeed");
        let var_result = pelt
            .detect(&data.view(), ChangeType::Variance)
            .expect("Variance detect should succeed");
        let mv_result = pelt
            .detect(&data.view(), ChangeType::MeanAndVariance)
            .expect("MeanAndVariance detect should succeed");

        assert!(mean_result.total_cost.is_finite());
        assert!(var_result.total_cost.is_finite());
        assert!(mv_result.total_cost.is_finite());
    }

    // -------------------------------------------------------------------------
    // Test 27: Empty data handling
    // -------------------------------------------------------------------------
    #[test]
    fn test_empty_data() {
        let data = Array1::<f64>::zeros(0);

        let cusum = Cusum::new(0.0, 5.0, 0.5);
        let result = cusum
            .detect(&data.view())
            .expect("CUSUM empty should succeed");
        assert!(result.locations.is_empty());

        let bayes = BayesianChangePoint::new(0.1, 0.5);
        let result = bayes
            .detect(&data.view())
            .expect("Bayes empty should succeed");
        assert!(result.locations.is_empty());
    }
}
