//! Streaming/Online Statistics Module for NumRS2
//!
//! This module provides online algorithms that compute statistics in a single pass
//! without storing all data points. This is critical for large datasets and real-time
//! applications where memory constraints or continuous data streams make batch
//! processing infeasible.
//!
//! # Algorithms
//!
//! - **Welford's algorithm**: Numerically stable online mean and variance computation
//! - **Parallel Welford**: Online covariance and Pearson correlation
//! - **Exponential Moving Average (EMA)**: Weighted recent-value smoothing
//! - **P-squared algorithm**: Online quantile estimation without storing data
//! - **Online Histogram**: Fixed-bin frequency counting
//!
//! # Mathematical References
//!
//! - Welford, B. P. (1962). "Note on a method for calculating corrected sums of
//!   squares and products." *Technometrics*, 4(3), 419-420.
//! - Chan, T. F., Golub, G. H., & LeVeque, R. J. (1979). "Updating formulae and
//!   a pairwise algorithm for computing sample variances."
//! - Jain, R., & Chlamtac, I. (1985). "The P-squared algorithm for dynamic
//!   calculation of quantiles and histograms without storing observations."
//!   *Communications of the ACM*, 28(10), 1076-1085.
//!
//! # Design
//!
//! All structs are `Send + Sync` and support merging for parallel-friendly computation.
//! No `unwrap()` is used in production code. Methods return `Option` or `Result` when
//! minimum data requirements are not met.

use crate::error::{NumRs2Error, Result};

// ─────────────────────────────────────────────────────────────────────────────
// OnlineStats: Welford's Algorithm for Mean/Variance
// ─────────────────────────────────────────────────────────────────────────────

/// Online computation of mean, variance, standard deviation, min, and max
/// using Welford's numerically stable algorithm.
///
/// # Example
///
/// ```
/// use numrs2::new_modules::streaming_stats::OnlineStats;
///
/// let mut stats = OnlineStats::new();
/// for x in &[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
///     stats.push(*x);
/// }
/// assert!((stats.mean().unwrap() - 5.0).abs() < 1e-10);
/// assert!((stats.variance().unwrap() - 4.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct OnlineStats {
    count: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineStats {
    /// Creates a new empty `OnlineStats` accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Incorporates a new observation into the running statistics.
    ///
    /// Uses Welford's algorithm for numerically stable variance computation.
    pub fn push(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Returns the number of observations seen so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the running mean, or `None` if no observations have been pushed.
    pub fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.mean)
        }
    }

    /// Returns the population variance (biased), or `None` if no observations.
    ///
    /// Population variance = M2 / n
    pub fn variance(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.m2 / self.count as f64)
        }
    }

    /// Returns the sample variance (unbiased, Bessel-corrected), or `None` if
    /// fewer than 2 observations have been pushed.
    ///
    /// Sample variance = M2 / (n - 1)
    pub fn sample_variance(&self) -> Option<f64> {
        if self.count < 2 {
            None
        } else {
            Some(self.m2 / (self.count - 1) as f64)
        }
    }

    /// Returns the population standard deviation, or `None` if no observations.
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    /// Returns the sample standard deviation, or `None` if fewer than 2 observations.
    pub fn sample_std_dev(&self) -> Option<f64> {
        self.sample_variance().map(|v| v.sqrt())
    }

    /// Returns the minimum observed value, or `None` if no observations.
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.min)
        }
    }

    /// Returns the maximum observed value, or `None` if no observations.
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.max)
        }
    }

    /// Merges another `OnlineStats` accumulator into this one.
    ///
    /// This enables parallel computation: split data across threads, compute
    /// partial `OnlineStats`, then merge results. Uses Chan et al.'s parallel
    /// algorithm for combining partial aggregates.
    pub fn merge(&mut self, other: &OnlineStats) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        let combined_mean = self.mean + delta * (other.count as f64 / combined_count as f64);

        // Chan's parallel formula: M2_combined = M2_a + M2_b + delta^2 * n_a * n_b / n_combined
        let combined_m2 = self.m2
            + other.m2
            + delta * delta * (self.count as f64) * (other.count as f64) / (combined_count as f64);

        self.count = combined_count;
        self.mean = combined_mean;
        self.m2 = combined_m2;

        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineCovariance: Parallel Welford for Covariance/Correlation
// ─────────────────────────────────────────────────────────────────────────────

/// Online computation of covariance and Pearson correlation coefficient
/// between two variables using the parallel Welford algorithm.
///
/// # Example
///
/// ```
/// use numrs2::new_modules::streaming_stats::OnlineCovariance;
///
/// let mut cov = OnlineCovariance::new();
/// // Perfect positive correlation
/// for i in 0..100 {
///     cov.push(i as f64, 2.0 * i as f64 + 1.0);
/// }
/// assert!((cov.correlation().unwrap() - 1.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct OnlineCovariance {
    count: u64,
    mean_x: f64,
    mean_y: f64,
    m2_x: f64,
    m2_y: f64,
    co_moment: f64,
}

impl Default for OnlineCovariance {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineCovariance {
    /// Creates a new empty covariance accumulator.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean_x: 0.0,
            mean_y: 0.0,
            m2_x: 0.0,
            m2_y: 0.0,
            co_moment: 0.0,
        }
    }

    /// Incorporates a new (x, y) observation pair.
    pub fn push(&mut self, x: f64, y: f64) {
        self.count += 1;
        let n = self.count as f64;

        let dx = x - self.mean_x;
        let dy = y - self.mean_y;

        self.mean_x += dx / n;
        self.mean_y += dy / n;

        // Note: use updated mean_x but old delta for y (Welford cross-moment)
        let dx2 = x - self.mean_x;
        let dy2 = y - self.mean_y;

        self.m2_x += dx * dx2;
        self.m2_y += dy * dy2;
        self.co_moment += dx * dy2;
    }

    /// Returns the number of observation pairs seen.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the mean of x values, or `None` if empty.
    pub fn mean_x(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.mean_x)
        }
    }

    /// Returns the mean of y values, or `None` if empty.
    pub fn mean_y(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.mean_y)
        }
    }

    /// Returns the population covariance, or `None` if no observations.
    ///
    /// Cov(X,Y) = co_moment / n
    pub fn covariance(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.co_moment / self.count as f64)
        }
    }

    /// Returns the sample covariance (Bessel-corrected), or `None` if fewer
    /// than 2 observations.
    pub fn sample_covariance(&self) -> Option<f64> {
        if self.count < 2 {
            None
        } else {
            Some(self.co_moment / (self.count - 1) as f64)
        }
    }

    /// Returns the Pearson correlation coefficient, or `None` if fewer than
    /// 2 observations or if either variable has zero variance.
    ///
    /// r = co_moment / sqrt(M2_x * M2_y)
    pub fn correlation(&self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        let denom = (self.m2_x * self.m2_y).sqrt();
        if denom < f64::EPSILON {
            return None;
        }
        Some(self.co_moment / denom)
    }

    /// Merges another `OnlineCovariance` into this one (parallel-friendly).
    pub fn merge(&mut self, other: &OnlineCovariance) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let combined_count = self.count + other.count;
        let dx = other.mean_x - self.mean_x;
        let dy = other.mean_y - self.mean_y;
        let n_a = self.count as f64;
        let n_b = other.count as f64;
        let n_ab = combined_count as f64;

        self.co_moment += other.co_moment + dx * dy * n_a * n_b / n_ab;
        self.m2_x += other.m2_x + dx * dx * n_a * n_b / n_ab;
        self.m2_y += other.m2_y + dy * dy * n_a * n_b / n_ab;
        self.mean_x += dx * n_b / n_ab;
        self.mean_y += dy * n_b / n_ab;
        self.count = combined_count;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExponentialMovingAverage
// ─────────────────────────────────────────────────────────────────────────────

/// Exponential Moving Average (EMA) filter.
///
/// Each new value is blended with the running average:
///   EMA_new = alpha * value + (1 - alpha) * EMA_old
///
/// The first pushed value initializes the EMA directly.
///
/// # Example
///
/// ```
/// use numrs2::new_modules::streaming_stats::ExponentialMovingAverage;
///
/// let mut ema = ExponentialMovingAverage::new(0.5).unwrap();
/// ema.push(10.0);
/// assert!((ema.value().unwrap() - 10.0).abs() < 1e-10);
/// ema.push(20.0);
/// assert!((ema.value().unwrap() - 15.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    alpha: f64,
    value: Option<f64>,
}

impl ExponentialMovingAverage {
    /// Creates a new EMA with the given smoothing factor `alpha`.
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha` is not in the interval (0, 1].
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(NumRs2Error::ValueError(format!(
                "alpha must be in (0, 1], got {}",
                alpha
            )));
        }
        Ok(Self { alpha, value: None })
    }

    /// Returns the smoothing factor alpha.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Pushes a new observation. The first value initializes the EMA; subsequent
    /// values are exponentially blended.
    pub fn push(&mut self, value: f64) {
        self.value = Some(match self.value {
            Some(prev) => self.alpha * value + (1.0 - self.alpha) * prev,
            None => value,
        });
    }

    /// Returns the current EMA value, or `None` if no values have been pushed.
    pub fn value(&self) -> Option<f64> {
        self.value
    }

    /// Resets the EMA to its initial empty state, preserving the alpha parameter.
    pub fn reset(&mut self) {
        self.value = None;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineQuantile: P-squared Algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Online quantile estimation using the P-squared (P^2) algorithm.
///
/// The P-squared algorithm estimates a single quantile from a data stream using
/// only 5 markers, requiring O(1) memory regardless of stream length. It adjusts
/// marker positions using piecewise parabolic interpolation.
///
/// # Reference
///
/// Jain, R., & Chlamtac, I. (1985). "The P^2 algorithm for dynamic calculation
/// of quantiles and histograms without storing observations."
/// *Communications of the ACM*, 28(10), 1076-1085.
///
/// # Example
///
/// ```
/// use numrs2::new_modules::streaming_stats::OnlineQuantile;
///
/// let mut oq = OnlineQuantile::new(0.5).unwrap(); // median
/// for i in 1..=1000 {
///     oq.push(i as f64);
/// }
/// let est = oq.quantile().unwrap();
/// assert!((est - 500.5).abs() < 10.0); // approximate
/// ```
#[derive(Debug, Clone)]
pub struct OnlineQuantile {
    /// The target quantile p in (0, 1).
    p: f64,
    /// Marker heights q[0..5].
    q: [f64; 5],
    /// Marker positions n[0..5] (1-indexed logical positions).
    n: [f64; 5],
    /// Desired marker positions n'[0..5].
    n_prime: [f64; 5],
    /// Increments for desired positions dn'[0..5].
    dn: [f64; 5],
    /// Number of observations seen so far.
    count: u64,
}

impl OnlineQuantile {
    /// Creates a new P-squared quantile estimator for the given quantile `p`.
    ///
    /// # Errors
    ///
    /// Returns an error if `p` is not in the open interval (0, 1).
    pub fn new(quantile: f64) -> Result<Self> {
        if quantile <= 0.0 || quantile >= 1.0 {
            return Err(NumRs2Error::ValueError(format!(
                "quantile must be in (0, 1), got {}",
                quantile
            )));
        }
        Ok(Self {
            p: quantile,
            q: [0.0; 5],
            n: [1.0, 2.0, 3.0, 4.0, 5.0],
            n_prime: [
                1.0,
                1.0 + 2.0 * quantile,
                1.0 + 4.0 * quantile,
                3.0 + 2.0 * quantile,
                5.0,
            ],
            dn: [0.0, quantile / 2.0, quantile, (1.0 + quantile) / 2.0, 1.0],
            count: 0,
        })
    }

    /// Pushes a new observation into the quantile estimator.
    pub fn push(&mut self, value: f64) {
        self.count += 1;

        // Phase 1: Collect the first 5 observations
        if self.count <= 5 {
            self.q[self.count as usize - 1] = value;
            if self.count == 5 {
                // Sort the initial 5 values
                self.q
                    .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
            return;
        }

        // Phase 2: P-squared update
        // Find cell k such that q[k-1] <= value < q[k]
        let k = if value < self.q[0] {
            self.q[0] = value;
            0
        } else if value < self.q[1] {
            0
        } else if value < self.q[2] {
            1
        } else if value < self.q[3] {
            2
        } else if value < self.q[4] {
            3
        } else {
            if value > self.q[4] {
                self.q[4] = value;
            }
            3
        };

        // Increment positions of markers k+1 through 4
        for i in (k + 1)..5 {
            self.n[i] += 1.0;
        }

        // Update desired positions
        for i in 0..5 {
            self.n_prime[i] += self.dn[i];
        }

        // Adjust marker heights using P-squared (piecewise parabolic) interpolation
        for i in 1..4 {
            let d = self.n_prime[i] - self.n[i];
            if (d >= 1.0 && (self.n[i + 1] - self.n[i]) > 1.0)
                || (d <= -1.0 && (self.n[i - 1] - self.n[i]) < -1.0)
            {
                let sign = if d > 0.0 { 1.0 } else { -1.0 };

                // Try parabolic interpolation
                let qi = self.parabolic(i, sign);

                if qi > self.q[i - 1] && qi < self.q[i + 1] {
                    self.q[i] = qi;
                } else {
                    // Fall back to linear interpolation
                    self.q[i] = self.linear(i, sign);
                }
                self.n[i] += sign;
            }
        }
    }

    /// Parabolic (P-squared) interpolation formula.
    fn parabolic(&self, i: usize, d: f64) -> f64 {
        let ni = self.n[i];
        let nim1 = self.n[i - 1];
        let nip1 = self.n[i + 1];
        let qi = self.q[i];
        let qim1 = self.q[i - 1];
        let qip1 = self.q[i + 1];

        qi + d / (nip1 - nim1)
            * ((ni - nim1 + d) * (qip1 - qi) / (nip1 - ni)
                + (nip1 - ni - d) * (qi - qim1) / (ni - nim1))
    }

    /// Linear interpolation fallback.
    fn linear(&self, i: usize, d: f64) -> f64 {
        let idx = if d > 0.0 { i + 1 } else { i - 1 };
        self.q[i] + d * (self.q[idx] - self.q[i]) / (self.n[idx] - self.n[i])
    }

    /// Returns the estimated quantile value, or `None` if fewer than 5
    /// observations have been pushed.
    ///
    /// For exactly 5 observations, the exact quantile from the sorted initial
    /// data is returned. For more, the P-squared estimate is returned.
    pub fn quantile(&self) -> Option<f64> {
        if self.count < 5 {
            return None;
        }
        Some(self.q[2])
    }

    /// Returns the number of observations seen.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the target quantile parameter `p`.
    pub fn target_quantile(&self) -> f64 {
        self.p
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OnlineHistogram: Fixed-bin Histogram
// ─────────────────────────────────────────────────────────────────────────────

/// Online fixed-bin histogram that accumulates frequency counts in a single pass.
///
/// Values outside the configured range [min, max] are counted in underflow/overflow
/// tallies but not placed into bins.
///
/// # Example
///
/// ```
/// use numrs2::new_modules::streaming_stats::OnlineHistogram;
///
/// let mut hist = OnlineHistogram::new(0.0, 10.0, 5).unwrap();
/// for v in &[1.0, 3.0, 5.0, 7.0, 9.0] {
///     hist.push(*v);
/// }
/// assert_eq!(hist.total(), 5);
/// assert_eq!(hist.bins().len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct OnlineHistogram {
    bins: Vec<(f64, f64, u64)>,
    bin_width: f64,
    range_min: f64,
    range_max: f64,
    underflow: u64,
    overflow: u64,
}

impl OnlineHistogram {
    /// Creates a new histogram with `num_bins` equal-width bins covering [min, max].
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `num_bins` is zero
    /// - `min >= max`
    /// - `min` or `max` is not finite
    pub fn new(min: f64, max: f64, num_bins: usize) -> Result<Self> {
        if num_bins == 0 {
            return Err(NumRs2Error::ValueError(
                "num_bins must be at least 1".to_string(),
            ));
        }
        if !min.is_finite() || !max.is_finite() {
            return Err(NumRs2Error::ValueError(
                "min and max must be finite".to_string(),
            ));
        }
        if min >= max {
            return Err(NumRs2Error::ValueError(format!(
                "min ({}) must be less than max ({})",
                min, max
            )));
        }

        let bin_width = (max - min) / num_bins as f64;
        let mut bins = Vec::with_capacity(num_bins);
        for i in 0..num_bins {
            let lo = min + i as f64 * bin_width;
            let hi = if i == num_bins - 1 {
                max
            } else {
                min + (i + 1) as f64 * bin_width
            };
            bins.push((lo, hi, 0u64));
        }

        Ok(Self {
            bins,
            bin_width,
            range_min: min,
            range_max: max,
            underflow: 0,
            overflow: 0,
        })
    }

    /// Pushes a value into the histogram.
    ///
    /// Values below `min` increment the underflow counter; values above `max`
    /// increment the overflow counter.
    pub fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        if value < self.range_min {
            self.underflow += 1;
            return;
        }
        if value > self.range_max {
            self.overflow += 1;
            return;
        }

        // Compute bin index
        let idx = ((value - self.range_min) / self.bin_width) as usize;
        // Clamp to last bin (handles value == range_max)
        let idx = if idx >= self.bins.len() {
            self.bins.len() - 1
        } else {
            idx
        };
        self.bins[idx].2 += 1;
    }

    /// Returns a slice of bins as (low_edge, high_edge, count) triples.
    pub fn bins(&self) -> &[(f64, f64, u64)] {
        &self.bins
    }

    /// Returns the total count of observations that fell within [min, max].
    pub fn total(&self) -> u64 {
        self.bins.iter().map(|(_, _, c)| c).sum()
    }

    /// Returns the total count including underflow and overflow.
    pub fn total_all(&self) -> u64 {
        self.total() + self.underflow + self.overflow
    }

    /// Returns the number of observations below the histogram range.
    pub fn underflow(&self) -> u64 {
        self.underflow
    }

    /// Returns the number of observations above the histogram range.
    pub fn overflow(&self) -> u64 {
        self.overflow
    }

    /// Returns the number of bins.
    pub fn num_bins(&self) -> usize {
        self.bins.len()
    }

    /// Returns the bin width.
    pub fn bin_width(&self) -> f64 {
        self.bin_width
    }

    /// Resets all bin counts and overflow/underflow counters to zero.
    pub fn reset(&mut self) {
        for bin in &mut self.bins {
            bin.2 = 0;
        }
        self.underflow = 0;
        self.overflow = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- OnlineStats tests ---

    #[test]
    fn test_streaming_online_stats_empty() {
        let stats = OnlineStats::new();
        assert_eq!(stats.count(), 0);
        assert!(stats.mean().is_none());
        assert!(stats.variance().is_none());
        assert!(stats.sample_variance().is_none());
        assert!(stats.std_dev().is_none());
        assert!(stats.min().is_none());
        assert!(stats.max().is_none());
    }

    #[test]
    fn test_streaming_online_stats_single_value() {
        let mut stats = OnlineStats::new();
        stats.push(42.0);
        assert_eq!(stats.count(), 1);
        assert!((stats.mean().expect("should have mean") - 42.0).abs() < 1e-10);
        assert!((stats.variance().expect("should have var") - 0.0).abs() < 1e-10);
        assert!(stats.sample_variance().is_none()); // need at least 2
        assert_eq!(stats.min(), Some(42.0));
        assert_eq!(stats.max(), Some(42.0));
    }

    #[test]
    fn test_streaming_online_stats_known_dataset() {
        // Dataset: [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5.0, Population variance = 4.0, Sample variance = 32/7
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut stats = OnlineStats::new();
        for &x in &data {
            stats.push(x);
        }

        assert_eq!(stats.count(), 8);
        assert!((stats.mean().expect("mean") - 5.0).abs() < 1e-10);
        assert!((stats.variance().expect("var") - 4.0).abs() < 1e-10);
        assert!((stats.sample_variance().expect("sample var") - 32.0 / 7.0).abs() < 1e-10);
        assert!((stats.std_dev().expect("std") - 2.0).abs() < 1e-10);
        assert_eq!(stats.min(), Some(2.0));
        assert_eq!(stats.max(), Some(9.0));
    }

    #[test]
    fn test_streaming_online_stats_merge() {
        let data_a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let data_b = [6.0, 7.0, 8.0, 9.0, 10.0];

        let mut stats_a = OnlineStats::new();
        for &x in &data_a {
            stats_a.push(x);
        }
        let mut stats_b = OnlineStats::new();
        for &x in &data_b {
            stats_b.push(x);
        }

        let mut stats_all = OnlineStats::new();
        for &x in data_a.iter().chain(data_b.iter()) {
            stats_all.push(x);
        }

        stats_a.merge(&stats_b);

        assert_eq!(stats_a.count(), stats_all.count());
        assert!((stats_a.mean().expect("m") - stats_all.mean().expect("m")).abs() < 1e-10);
        assert!((stats_a.variance().expect("v") - stats_all.variance().expect("v")).abs() < 1e-10);
        assert_eq!(stats_a.min(), stats_all.min());
        assert_eq!(stats_a.max(), stats_all.max());
    }

    #[test]
    fn test_streaming_online_stats_merge_empty() {
        let mut stats = OnlineStats::new();
        stats.push(5.0);
        let empty = OnlineStats::new();
        stats.merge(&empty);
        assert_eq!(stats.count(), 1);
        assert!((stats.mean().expect("m") - 5.0).abs() < 1e-10);

        let mut empty2 = OnlineStats::new();
        let mut non_empty = OnlineStats::new();
        non_empty.push(10.0);
        empty2.merge(&non_empty);
        assert_eq!(empty2.count(), 1);
        assert!((empty2.mean().expect("m") - 10.0).abs() < 1e-10);
    }

    // --- OnlineCovariance tests ---

    #[test]
    fn test_streaming_covariance_empty() {
        let cov = OnlineCovariance::new();
        assert_eq!(cov.count(), 0);
        assert!(cov.covariance().is_none());
        assert!(cov.correlation().is_none());
    }

    #[test]
    fn test_streaming_covariance_perfect_positive() {
        let mut cov = OnlineCovariance::new();
        for i in 0..100 {
            cov.push(i as f64, 2.0 * i as f64 + 1.0);
        }
        let r = cov.correlation().expect("should have correlation");
        assert!((r - 1.0).abs() < 1e-10, "expected r=1.0, got {}", r);
    }

    #[test]
    fn test_streaming_covariance_perfect_negative() {
        let mut cov = OnlineCovariance::new();
        for i in 0..100 {
            cov.push(i as f64, -(i as f64) + 50.0);
        }
        let r = cov.correlation().expect("should have correlation");
        assert!((r - (-1.0)).abs() < 1e-10, "expected r=-1.0, got {}", r);
    }

    #[test]
    fn test_streaming_covariance_known_values() {
        // X = [1, 2, 3, 4, 5], Y = [2, 4, 5, 4, 5]
        // Cov(X,Y) population = 1.0
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [2.0, 4.0, 5.0, 4.0, 5.0];
        let mut cov = OnlineCovariance::new();
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            cov.push(x, y);
        }
        assert_eq!(cov.count(), 5);
        assert!((cov.covariance().expect("cov") - 1.2).abs() < 1e-10);
        assert!((cov.sample_covariance().expect("scov") - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_covariance_merge() {
        let mut cov_a = OnlineCovariance::new();
        let mut cov_b = OnlineCovariance::new();
        let mut cov_all = OnlineCovariance::new();

        for i in 0..50 {
            let x = i as f64;
            let y = 3.0 * x + 2.0;
            cov_a.push(x, y);
            cov_all.push(x, y);
        }
        for i in 50..100 {
            let x = i as f64;
            let y = 3.0 * x + 2.0;
            cov_b.push(x, y);
            cov_all.push(x, y);
        }

        cov_a.merge(&cov_b);
        assert_eq!(cov_a.count(), cov_all.count());
        assert!((cov_a.covariance().expect("c") - cov_all.covariance().expect("c")).abs() < 1e-8);
        assert!(
            (cov_a.correlation().expect("r") - cov_all.correlation().expect("r")).abs() < 1e-10
        );
    }

    // --- ExponentialMovingAverage tests ---

    #[test]
    fn test_streaming_ema_invalid_alpha() {
        assert!(ExponentialMovingAverage::new(0.0).is_err());
        assert!(ExponentialMovingAverage::new(-0.1).is_err());
        assert!(ExponentialMovingAverage::new(1.5).is_err());
        assert!(ExponentialMovingAverage::new(1.0).is_ok());
        assert!(ExponentialMovingAverage::new(0.5).is_ok());
    }

    #[test]
    fn test_streaming_ema_basic() {
        let mut ema = ExponentialMovingAverage::new(0.5).expect("valid alpha");
        assert!(ema.value().is_none());

        ema.push(10.0);
        assert!((ema.value().expect("v") - 10.0).abs() < 1e-10);

        ema.push(20.0);
        // EMA = 0.5 * 20 + 0.5 * 10 = 15
        assert!((ema.value().expect("v") - 15.0).abs() < 1e-10);

        ema.push(30.0);
        // EMA = 0.5 * 30 + 0.5 * 15 = 22.5
        assert!((ema.value().expect("v") - 22.5).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_ema_alpha_one() {
        // alpha=1 means EMA always equals the latest value
        let mut ema = ExponentialMovingAverage::new(1.0).expect("valid");
        ema.push(5.0);
        ema.push(10.0);
        ema.push(100.0);
        assert!((ema.value().expect("v") - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_ema_reset() {
        let mut ema = ExponentialMovingAverage::new(0.3).expect("valid");
        ema.push(42.0);
        assert!(ema.value().is_some());
        ema.reset();
        assert!(ema.value().is_none());
        assert!((ema.alpha() - 0.3).abs() < 1e-10);
    }

    // --- OnlineQuantile (P-squared) tests ---

    #[test]
    fn test_streaming_quantile_invalid() {
        assert!(OnlineQuantile::new(0.0).is_err());
        assert!(OnlineQuantile::new(1.0).is_err());
        assert!(OnlineQuantile::new(-0.5).is_err());
        assert!(OnlineQuantile::new(0.5).is_ok());
    }

    #[test]
    fn test_streaming_quantile_insufficient_data() {
        let mut oq = OnlineQuantile::new(0.5).expect("valid");
        oq.push(1.0);
        oq.push(2.0);
        oq.push(3.0);
        oq.push(4.0);
        assert!(oq.quantile().is_none()); // need 5
    }

    #[test]
    fn test_streaming_quantile_median_uniform() {
        let mut oq = OnlineQuantile::new(0.5).expect("valid");
        // Push 1..=1000
        for i in 1..=1000 {
            oq.push(i as f64);
        }
        let est = oq.quantile().expect("should have estimate");
        // True median of 1..=1000 is 500.5
        assert!(
            (est - 500.5).abs() < 30.0,
            "Median estimate {} too far from 500.5",
            est
        );
    }

    #[test]
    fn test_streaming_quantile_90th_percentile() {
        let mut oq = OnlineQuantile::new(0.9).expect("valid");
        for i in 1..=10000 {
            oq.push(i as f64);
        }
        let est = oq.quantile().expect("estimate");
        // True 90th percentile ~ 9000.5
        assert!(
            (est - 9000.5).abs() < 200.0,
            "P90 estimate {} too far from 9000.5",
            est
        );
    }

    // --- OnlineHistogram tests ---

    #[test]
    fn test_streaming_histogram_invalid() {
        assert!(OnlineHistogram::new(10.0, 5.0, 10).is_err()); // min > max
        assert!(OnlineHistogram::new(0.0, 10.0, 0).is_err()); // zero bins
        assert!(OnlineHistogram::new(f64::NAN, 10.0, 5).is_err());
        assert!(OnlineHistogram::new(0.0, f64::INFINITY, 5).is_err());
    }

    #[test]
    fn test_streaming_histogram_basic() {
        let mut hist = OnlineHistogram::new(0.0, 10.0, 5).expect("valid");
        // Bins: [0,2), [2,4), [4,6), [6,8), [8,10]
        hist.push(1.0); // bin 0
        hist.push(3.0); // bin 1
        hist.push(5.0); // bin 2
        hist.push(7.0); // bin 3
        hist.push(9.0); // bin 4

        assert_eq!(hist.total(), 5);
        assert_eq!(hist.num_bins(), 5);
        let bins = hist.bins();
        assert_eq!(bins[0].2, 1);
        assert_eq!(bins[1].2, 1);
        assert_eq!(bins[2].2, 1);
        assert_eq!(bins[3].2, 1);
        assert_eq!(bins[4].2, 1);
    }

    #[test]
    fn test_streaming_histogram_overflow_underflow() {
        let mut hist = OnlineHistogram::new(0.0, 10.0, 5).expect("valid");
        hist.push(-1.0); // underflow
        hist.push(11.0); // overflow
        hist.push(5.0); // in range

        assert_eq!(hist.underflow(), 1);
        assert_eq!(hist.overflow(), 1);
        assert_eq!(hist.total(), 1);
        assert_eq!(hist.total_all(), 3);
    }

    #[test]
    fn test_streaming_histogram_edge_value() {
        let mut hist = OnlineHistogram::new(0.0, 10.0, 5).expect("valid");
        hist.push(0.0); // should go to bin 0
        hist.push(10.0); // should go to last bin (edge case)

        assert_eq!(hist.total(), 2);
        assert_eq!(hist.bins()[0].2, 1);
        assert_eq!(hist.bins()[4].2, 1);
    }

    #[test]
    fn test_streaming_histogram_reset() {
        let mut hist = OnlineHistogram::new(0.0, 10.0, 5).expect("valid");
        hist.push(1.0);
        hist.push(3.0);
        hist.push(-1.0);
        hist.reset();

        assert_eq!(hist.total(), 0);
        assert_eq!(hist.underflow(), 0);
        assert_eq!(hist.overflow(), 0);
    }

    #[test]
    fn test_streaming_histogram_nan_ignored() {
        let mut hist = OnlineHistogram::new(0.0, 10.0, 5).expect("valid");
        hist.push(f64::NAN);
        assert_eq!(hist.total_all(), 0);
    }

    // --- Numerical stability test ---

    #[test]
    fn test_streaming_online_stats_numerical_stability() {
        // Test Welford's stability with large offset
        let mut stats = OnlineStats::new();
        let offset = 1e9;
        for i in 0..1000 {
            stats.push(offset + i as f64);
        }
        // Variance of 0..999 = (999*1000)/(12) = 83250 (population)
        // Actually: Var = (n^2 - 1)/12 for uniform 0..n-1
        // Var(0..999) = (1000^2 - 1)/12 = 999999/12 = 83333.25
        let expected_var = (1000.0_f64 * 1000.0 - 1.0) / 12.0;
        let var = stats.variance().expect("var");
        assert!(
            (var - expected_var).abs() < 0.01,
            "Variance {} too far from expected {}",
            var,
            expected_var
        );
    }
}
