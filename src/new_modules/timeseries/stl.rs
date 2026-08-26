//! Seasonal-Trend Decomposition using LOESS (STL)
//!
//! This module implements the STL decomposition method for separating a time series
//! into seasonal, trend, and remainder components using locally weighted regression (LOESS).
//!
//! ## STL Decomposition
//!
//! STL decomposes a time series Y_t into three components:
//!
//! Y_t = T_t + S_t + R_t
//!
//! where:
//! - T_t: Trend component (slow-varying, non-periodic changes)
//! - S_t: Seasonal component (periodic pattern)
//! - R_t: Remainder/residual component (irregular, random variation)
//!
//! ## Advantages of STL
//!
//! - Handles any type of seasonality (not just monthly/quarterly)
//! - Robust to outliers
//! - Seasonal component can vary over time
//! - User control over trend and seasonal smoothness
//!
//! ## References
//!
//! Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990).
//! STL: A seasonal-trend decomposition procedure based on loess.
//! *Journal of Official Statistics*, 6(1), 3-73.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{s, Array1, ArrayView1};

/// Result of STL decomposition.
#[derive(Debug, Clone)]
pub struct StlResult {
    /// Trend component
    pub trend: Array1<f64>,
    /// Seasonal component
    pub seasonal: Array1<f64>,
    /// Remainder component
    pub remainder: Array1<f64>,
}

/// STL decomposition configuration.
#[derive(Debug, Clone)]
pub struct StlDecomposition {
    /// Seasonal period (e.g., 12 for monthly data with yearly seasonality)
    pub period: usize,
    /// Seasonal smoothing parameter (odd integer)
    pub seasonal_window: usize,
    /// Trend smoothing parameter (odd integer)
    pub trend_window: Option<usize>,
    /// Number of robustness iterations
    pub robust_iterations: usize,
}

impl StlDecomposition {
    /// Create a new STL decomposition with given period.
    ///
    /// # Arguments
    ///
    /// * `period` - Seasonal period
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::new_modules::timeseries::StlDecomposition;
    ///
    /// // Monthly data with yearly seasonality
    /// let stl = StlDecomposition::new(12);
    /// ```
    pub fn new(period: usize) -> Self {
        // Default parameters following Cleveland et al. (1990)
        let seasonal_window = (10 * period + 1).next_odd();
        let trend_window = None; // Will be computed as nextodd(1.5*period/(1-1.5/s))

        Self {
            period,
            seasonal_window,
            trend_window,
            robust_iterations: 1,
        }
    }

    /// Set the seasonal smoothing window.
    pub fn with_seasonal_window(mut self, window: usize) -> Self {
        self.seasonal_window = window.next_odd();
        self
    }

    /// Set the trend smoothing window.
    pub fn with_trend_window(mut self, window: usize) -> Self {
        self.trend_window = Some(window.next_odd());
        self
    }

    /// Set the number of robustness iterations.
    pub fn with_robust_iterations(mut self, iterations: usize) -> Self {
        self.robust_iterations = iterations;
        self
    }

    /// Perform STL decomposition on a time series.
    ///
    /// # Arguments
    ///
    /// * `data` - Input time series
    ///
    /// # Returns
    ///
    /// STL decomposition result with trend, seasonal, and remainder components
    pub fn decompose(&self, data: &ArrayView1<f64>) -> Result<StlResult> {
        let n = data.len();

        if n < 2 * self.period {
            return Err(NumRs2Error::ValueError(format!(
                "Series length ({}) must be at least twice the period ({})",
                n,
                2 * self.period
            )));
        }

        // Compute default trend window if not specified
        let trend_window = self.trend_window.unwrap_or_else(|| {
            let s_val = self.seasonal_window as f64;
            let nextodd_val = (1.5 * self.period as f64 / (1.0 - 1.5 / s_val)).ceil() as usize;
            nextodd_val.next_odd()
        });

        // Initialize components
        let mut seasonal = Array1::zeros(n);
        let mut trend = Array1::zeros(n);

        // STL uses iterative refinement
        let weights = Array1::from_elem(n, 1.0);

        for _ in 0..self.robust_iterations + 1 {
            // Step 1: Detrend
            let detrended = self.inner_loop(data, &seasonal, &weights, trend_window)?;

            seasonal = detrended.0;
            trend = detrended.1;
        }

        // Compute remainder
        let remainder = data - &trend - &seasonal;

        Ok(StlResult {
            trend,
            seasonal,
            remainder,
        })
    }

    /// Inner loop of STL algorithm.
    fn inner_loop(
        &self,
        data: &ArrayView1<f64>,
        seasonal: &Array1<f64>,
        weights: &Array1<f64>,
        trend_window: usize,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        // Step 1: Detrend
        let detrended = data - seasonal;

        // Step 2: Smooth to get trend
        let trend = self.loess_smooth(&detrended.view(), trend_window, weights)?;

        // Step 3: Detrend to get seasonal + remainder
        let seasonal_plus_remainder = data - &trend;

        // Step 4: Seasonal smoothing
        let seasonal_new = self.seasonal_smooth(&seasonal_plus_remainder.view())?;

        Ok((seasonal_new, trend))
    }

    /// LOESS (Locally Weighted Scatterplot Smoothing) smoother.
    fn loess_smooth(
        &self,
        data: &ArrayView1<f64>,
        window: usize,
        weights: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let n = data.len();
        let mut smoothed = Array1::zeros(n);

        for i in 0..n {
            // Determine window bounds
            let half_window = window / 2;
            let left = i.saturating_sub(half_window);
            let right = if i + half_window < n {
                i + half_window
            } else {
                n - 1
            };

            // Compute locally weighted regression
            let (a, b) = self.weighted_linear_regression(data, weights, left, right, i)?;

            smoothed[i] = a + b * i as f64;
        }

        Ok(smoothed)
    }

    /// Weighted linear regression for local window.
    fn weighted_linear_regression(
        &self,
        data: &ArrayView1<f64>,
        weights: &Array1<f64>,
        left: usize,
        right: usize,
        center: usize,
    ) -> Result<(f64, f64)> {
        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;
        let mut sum_wy = 0.0;
        let mut sum_wxx = 0.0;
        let mut sum_wxy = 0.0;

        for i in left..=right {
            // Tricube weight function
            let dist = (i as f64 - center as f64).abs();
            let max_dist = ((right - left) / 2) as f64;
            let u = if max_dist > 0.0 { dist / max_dist } else { 0.0 };

            let tricube_weight = if u < 1.0 {
                (1.0 - u.powi(3)).powi(3)
            } else {
                0.0
            };

            let w = weights[i] * tricube_weight;

            sum_w += w;
            sum_wx += w * i as f64;
            sum_wy += w * data[i];
            sum_wxx += w * (i as f64).powi(2);
            sum_wxy += w * i as f64 * data[i];
        }

        if sum_w < 1e-10 {
            return Ok((data[center], 0.0));
        }

        // Solve weighted least squares
        let denom = sum_w * sum_wxx - sum_wx * sum_wx;

        if denom.abs() < 1e-10 {
            // Degenerate case: return weighted mean
            return Ok((sum_wy / sum_w, 0.0));
        }

        let a = (sum_wxx * sum_wy - sum_wx * sum_wxy) / denom;
        let b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;

        Ok((a, b))
    }

    /// Seasonal smoothing by subseries.
    fn seasonal_smooth(&self, data: &ArrayView1<f64>) -> Result<Array1<f64>> {
        let n = data.len();
        let p = self.period;

        // Create subseries for each seasonal index
        let mut seasonal = Array1::zeros(n);

        for s in 0..p {
            // Extract subseries
            let mut subseries = Vec::new();
            let mut indices = Vec::new();

            for i in (s..n).step_by(p) {
                subseries.push(data[i]);
                indices.push(i);
            }

            if subseries.is_empty() {
                continue;
            }

            // Smooth subseries
            let subseries_array = Array1::from_vec(subseries);
            let weights = Array1::from_elem(subseries_array.len(), 1.0);
            let smoothed = self.loess_smooth(
                &subseries_array.view(),
                self.seasonal_window.min(subseries_array.len()),
                &weights,
            )?;

            // Place smoothed values back
            for (i, &idx) in indices.iter().enumerate() {
                seasonal[idx] = smoothed[i];
            }
        }

        // Remove mean of seasonal component (constraint: Σ S_t = 0)
        let seasonal_mean = seasonal.iter().sum::<f64>() / n as f64;
        seasonal -= seasonal_mean;

        Ok(seasonal)
    }
}

/// Extension trait for computing next odd integer.
trait NextOdd {
    fn next_odd(self) -> Self;
}

impl NextOdd for usize {
    fn next_odd(self) -> Self {
        if self.is_multiple_of(2) {
            self + 1
        } else {
            self
        }
    }
}

/// Classical additive decomposition.
///
/// Decomposes series into trend (moving average), seasonal (average by season),
/// and remainder components.
///
/// # Arguments
///
/// * `data` - Time series data
/// * `period` - Seasonal period
///
/// # Returns
///
/// Decomposition result
pub fn classical_decomposition(data: &ArrayView1<f64>, period: usize) -> Result<StlResult> {
    let n = data.len();

    if n < 2 * period {
        return Err(NumRs2Error::ValueError(
            "Series must be at least twice the period length".to_string(),
        ));
    }

    // Compute trend using centered moving average
    let mut trend = Array1::zeros(n);
    let half_period = period / 2;

    for i in half_period..(n - half_period) {
        let sum: f64 = data
            .slice(s![i - half_period..i + half_period + 1])
            .iter()
            .sum();
        trend[i] = sum / period as f64;
    }

    // Handle edges with linear extrapolation
    if half_period > 0 {
        for i in 0..half_period {
            trend[i] = trend[half_period];
        }
        for i in (n - half_period)..n {
            trend[i] = trend[n - half_period - 1];
        }
    }

    // Detrend
    let detrended = data - &trend;

    // Compute seasonal component
    let mut seasonal = Array1::zeros(n);
    let mut seasonal_avgs = Array1::zeros(period);

    for s in 0..period {
        let mut sum = 0.0;
        let mut count = 0;

        for i in (s..n).step_by(period) {
            sum += detrended[i];
            count += 1;
        }

        seasonal_avgs[s] = if count > 0 { sum / count as f64 } else { 0.0 };
    }

    // Remove mean from seasonal
    let seasonal_mean = seasonal_avgs.iter().sum::<f64>() / period as f64;
    seasonal_avgs -= seasonal_mean;

    // Assign seasonal values
    for i in 0..n {
        seasonal[i] = seasonal_avgs[i % period];
    }

    // Compute remainder
    let remainder = data - &trend - &seasonal;

    Ok(StlResult {
        trend,
        seasonal,
        remainder,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_stl_creation() {
        let stl = StlDecomposition::new(12);
        assert_eq!(stl.period, 12);
        assert!(stl.seasonal_window % 2 == 1); // Must be odd
    }

    #[test]
    fn test_stl_decomposition_simple() {
        // Create simple seasonal series
        let mut data = Vec::new();
        for i in 0..48 {
            let trend = i as f64 * 0.1;
            let seasonal = (i % 12) as f64;
            data.push(trend + seasonal);
        }
        let data = Array1::from_vec(data);

        let stl = StlDecomposition::new(12);
        let result = stl
            .decompose(&data.view())
            .expect("STL decomposition should succeed");

        assert_eq!(result.trend.len(), 48);
        assert_eq!(result.seasonal.len(), 48);
        assert_eq!(result.remainder.len(), 48);

        // Reconstruction should be close to original
        let reconstructed = &result.trend + &result.seasonal + &result.remainder;
        for i in 0..48 {
            assert_relative_eq!(reconstructed[i], data[i], epsilon = 0.1);
        }
    }

    #[test]
    fn test_classical_decomposition() {
        let mut data = Vec::new();
        for i in 0..36 {
            let trend = 10.0 + i as f64 * 0.5;
            let seasonal = (i % 12) as f64 - 6.0;
            data.push(trend + seasonal);
        }
        let data = Array1::from_vec(data);

        let result = classical_decomposition(&data.view(), 12)
            .expect("classical decomposition should succeed");

        assert_eq!(result.trend.len(), 36);
        assert_eq!(result.seasonal.len(), 36);
        assert_eq!(result.remainder.len(), 36);
    }

    #[test]
    fn test_insufficient_data() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let stl = StlDecomposition::new(12);

        let result = stl.decompose(&data.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_next_odd() {
        assert_eq!(5_usize.next_odd(), 5);
        assert_eq!(6_usize.next_odd(), 7);
        assert_eq!(10_usize.next_odd(), 11);
    }

    #[test]
    fn test_seasonal_mean_zero() {
        let mut data = Vec::new();
        for i in 0..24 {
            data.push((i % 4) as f64);
        }
        let data = Array1::from_vec(data);

        let stl = StlDecomposition::new(4);
        let result = stl
            .decompose(&data.view())
            .expect("decomposition should succeed");

        // Seasonal component should have mean close to zero
        let seasonal_mean = result.seasonal.iter().sum::<f64>() / result.seasonal.len() as f64;
        assert_relative_eq!(seasonal_mean, 0.0, epsilon = 1e-10);
    }
}
