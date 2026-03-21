//! Statistical plotting module
//!
//! This module provides statistical visualization capabilities including
//! histograms, box plots, violin plots, scatter matrices, and correlation heatmaps.

use super::*;
use plotters::prelude::*;
use plotters::style::Color;
use scirs2_core::ndarray::{Array1, Array2};
use std::path::Path;

// Helper functions using scirs2_stats public API
fn mean(data: &Array1<f64>) -> Result<f64, String> {
    if data.is_empty() {
        return Err("Empty array".to_string());
    }
    Ok(data.sum() / data.len() as f64)
}

fn median(data: &Array1<f64>) -> Result<f64, String> {
    if data.is_empty() {
        return Err("Empty array".to_string());
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    Ok(if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    })
}

fn percentile(data: &Array1<f64>, p: f64) -> Result<f64, String> {
    if data.is_empty() {
        return Err("Empty array".to_string());
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    Ok(sorted[idx.min(sorted.len() - 1)])
}

fn std_dev(data: &Array1<f64>) -> Result<f64, String> {
    if data.is_empty() {
        return Err("Empty array".to_string());
    }
    let mean_val = mean(data)?;
    let variance = data.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / data.len() as f64;
    Ok(variance.sqrt())
}

/// Statistical plot structure
pub struct StatPlot {
    config: PlotConfig,
}

/// Binning strategy for histograms
#[derive(Debug, Clone, Copy)]
pub enum BinStrategy {
    /// Automatic binning (Sturges' formula)
    Auto,
    /// Sturges' formula: k = ceil(log2(n)) + 1
    Sturges,
    /// Scott's rule: h = 3.5 * σ / n^(1/3)
    Scott,
    /// Freedman-Diaconis rule: h = 2 * IQR / n^(1/3)
    FreedmanDiaconis,
    /// Fixed number of bins
    Fixed(usize),
}

impl StatPlot {
    /// Create a new statistical plot
    pub fn new(config: PlotConfig) -> Self {
        Self { config }
    }

    /// Create a histogram with the specified binning strategy
    pub fn histogram(
        &self,
        data: &Array1<f64>,
        strategy: BinStrategy,
        path: &Path,
    ) -> VizResult<()> {
        let bins = self.compute_bins(data, strategy)?;
        let (counts, edges) = compute_histogram_with_edges(data, bins)?;

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        let max_count = counts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(edges[0]..edges[edges.len() - 1], 0.0..max_count * 1.1)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .x_desc(&self.config.x_axis.label)
            .y_desc("Frequency")
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw histogram bars
        for i in 0..counts.len() {
            let x0 = edges[i];
            let x1 = edges[i + 1];
            let y = counts[i];

            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(x0, 0.0), (x1, y)],
                    BLUE.mix(0.6).filled(),
                )))
                .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;
        }

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }

    /// Create a box plot
    pub fn boxplot(&self, data: &Array1<f64>, path: &Path) -> VizResult<()> {
        let stats = compute_box_stats(data)?;

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        let y_min = stats.min - (stats.max - stats.min) * 0.1;
        let y_max = stats.max + (stats.max - stats.min) * 0.1;

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..2.0, y_min..y_max)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .y_desc(&self.config.y_axis.label)
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw box
        let x_center = 1.0;
        let box_width = 0.3;

        // Box (Q1 to Q3)
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [
                    (x_center - box_width / 2.0, stats.q1),
                    (x_center + box_width / 2.0, stats.q3),
                ],
                BLUE.mix(0.3).filled(),
            )))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Median line
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![
                    (x_center - box_width / 2.0, stats.median),
                    (x_center + box_width / 2.0, stats.median),
                ],
                RED,
            )))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Whiskers
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(x_center, stats.q1), (x_center, stats.min)],
                BLACK,
            )))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(x_center, stats.q3), (x_center, stats.max)],
                BLACK,
            )))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }

    /// Create a Q-Q plot (quantile-quantile)
    pub fn qqplot(&self, data: &Array1<f64>, path: &Path) -> VizResult<()> {
        let sorted_data = {
            let mut v = data.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            v
        };

        let n = sorted_data.len();
        let theoretical_quantiles: Vec<f64> = (0..n)
            .map(|i| {
                let p = (i as f64 + 0.5) / n as f64;
                normal_quantile(p)
            })
            .collect();

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        let x_min = theoretical_quantiles
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let x_max = theoretical_quantiles
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = sorted_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = sorted_data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .x_desc("Theoretical Quantiles")
            .y_desc("Sample Quantiles")
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Plot points
        chart
            .draw_series(
                theoretical_quantiles
                    .iter()
                    .zip(sorted_data.iter())
                    .map(|(&x, &y)| Circle::new((x, y), 3, BLUE.filled())),
            )
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw reference line (y = x)
        chart
            .draw_series(std::iter::once(PathElement::new(
                vec![(x_min, y_min), (x_max, y_max)],
                RED,
            )))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }

    /// Compute number of bins based on strategy
    fn compute_bins(&self, data: &Array1<f64>, strategy: BinStrategy) -> VizResult<usize> {
        let n = data.len();
        if n == 0 {
            return Err(VizError::InvalidData("Empty data".to_string()));
        }

        match strategy {
            BinStrategy::Auto | BinStrategy::Sturges => {
                Ok(((n as f64).log2().ceil() as usize + 1).max(1))
            }
            BinStrategy::Scott => {
                let sigma = std_dev(data).map_err(|e| VizError::InvalidData(format!("{:?}", e)))?;
                let h = 3.5 * sigma / (n as f64).powf(1.0 / 3.0);
                let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let bins = ((max - min) / h).ceil() as usize;
                Ok(bins.max(1))
            }
            BinStrategy::FreedmanDiaconis => {
                let q1 = percentile(data, 25.0)
                    .map_err(|e| VizError::InvalidData(format!("{:?}", e)))?;
                let q3 = percentile(data, 75.0)
                    .map_err(|e| VizError::InvalidData(format!("{:?}", e)))?;
                let iqr = q3 - q1;
                let h = 2.0 * iqr / (n as f64).powf(1.0 / 3.0);
                if h <= 0.0 {
                    return Ok(((n as f64).log2().ceil() as usize + 1).max(1));
                }
                let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let bins = ((max - min) / h).ceil() as usize;
                Ok(bins.max(1))
            }
            BinStrategy::Fixed(bins) => {
                if bins == 0 {
                    Err(VizError::InvalidConfig(
                        "Number of bins must be positive".to_string(),
                    ))
                } else {
                    Ok(bins)
                }
            }
        }
    }
}

/// Box plot statistics
struct BoxStats {
    min: f64,
    q1: f64,
    median: f64,
    q3: f64,
    max: f64,
}

/// Compute box plot statistics
fn compute_box_stats(data: &Array1<f64>) -> VizResult<BoxStats> {
    if data.is_empty() {
        return Err(VizError::InvalidData("Empty data".to_string()));
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let median_val = median(data).map_err(|e| VizError::InvalidData(format!("{:?}", e)))?;
    let q1 = percentile(data, 25.0).map_err(|e| VizError::InvalidData(format!("{:?}", e)))?;
    let q3 = percentile(data, 75.0).map_err(|e| VizError::InvalidData(format!("{:?}", e)))?;

    Ok(BoxStats {
        min,
        q1,
        median: median_val,
        q3,
        max,
    })
}

/// Compute histogram with explicit bin edges
fn compute_histogram_with_edges(
    data: &Array1<f64>,
    bins: usize,
) -> VizResult<(Vec<f64>, Vec<f64>)> {
    if data.is_empty() {
        return Err(VizError::InvalidData("Empty data".to_string()));
    }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if !min.is_finite() || !max.is_finite() {
        return Err(VizError::InvalidData(
            "Non-finite values in data".to_string(),
        ));
    }

    let bin_width = (max - min) / bins as f64;
    let edges: Vec<f64> = (0..=bins).map(|i| min + i as f64 * bin_width).collect();

    let mut counts = vec![0.0; bins];
    for &val in data.iter() {
        if val.is_finite() {
            let bin_idx = ((val - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            counts[bin_idx] += 1.0;
        }
    }

    Ok((counts, edges))
}

/// Compute the inverse of the standard normal CDF (approximate)
fn normal_quantile(p: f64) -> f64 {
    // Approximate inverse normal CDF using rational approximation
    // This is a simplified version; for production use, consider a more accurate implementation
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for rational approximation
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Lower region
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_bin_computation() {
        let plot = StatPlot::new(PlotConfig::default());
        let data = Array1::from_vec((0..100).map(|x| x as f64).collect());

        let bins = plot.compute_bins(&data, BinStrategy::Sturges);
        assert!(bins.is_ok());
        assert!(bins.unwrap_or(0) > 0);
    }

    #[test]
    fn test_box_stats() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = compute_box_stats(&data);
        assert!(stats.is_ok());

        let stats = stats.unwrap_or_else(|_| panic!("Failed to compute stats"));
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_normal_quantile() {
        let q = normal_quantile(0.5);
        assert!((q - 0.0).abs() < 0.01); // Should be close to 0

        let q = normal_quantile(0.975);
        assert!((q - 1.96).abs() < 0.05); // Should be close to 1.96
    }
}
