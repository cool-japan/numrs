//! Performance visualization module
//!
//! This module provides visualization for benchmark results, speedup curves,
//! and scaling analysis.

use super::*;
use plotters::prelude::*;
use plotters::style::Color;
use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::path::Path;

/// Performance plot structure
pub struct PerfPlot {
    config: PlotConfig,
}

/// Benchmark result data
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// Execution time (seconds)
    pub time: f64,
    /// Standard deviation
    pub std_dev: Option<f64>,
}

/// Scaling data point
#[derive(Debug, Clone)]
pub struct ScalingPoint {
    /// Number of cores/workers
    pub cores: usize,
    /// Execution time
    pub time: f64,
}

impl PerfPlot {
    /// Create a new performance plot
    pub fn new(config: PlotConfig) -> Self {
        Self { config }
    }

    /// Plot benchmark comparison (bar chart)
    pub fn benchmark_comparison(&self, results: &[BenchmarkResult], path: &Path) -> VizResult<()> {
        if results.is_empty() {
            return Err(VizError::InvalidData("No benchmark results".to_string()));
        }

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        let max_time = results
            .iter()
            .map(|r| r.time)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(80)
            .y_label_area_size(50)
            .build_cartesian_2d(0..results.len(), 0.0..(max_time * 1.1))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .y_desc("Time (seconds)")
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw bars
        for (idx, result) in results.iter().enumerate() {
            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(idx, 0.0), (idx + 1, result.time)],
                    BLUE.mix(0.6).filled(),
                )))
                .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

            // Note: error bars would require coordinate conversion, skipping for simplicity
        }

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }

    /// Plot speedup curve
    pub fn speedup_curve(
        &self,
        scaling_data: &[ScalingPoint],
        baseline: f64,
        path: &Path,
    ) -> VizResult<()> {
        if scaling_data.is_empty() {
            return Err(VizError::InvalidData("No scaling data".to_string()));
        }

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Compute speedups
        let speedups: Vec<(usize, f64)> = scaling_data
            .iter()
            .map(|p| (p.cores, baseline / p.time))
            .collect();

        let max_cores = speedups.iter().map(|(c, _)| *c).max().unwrap_or(1);
        let max_speedup = speedups
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..(max_cores as f64 * 1.1), 0.0..(max_speedup * 1.1))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .x_desc("Number of Cores")
            .y_desc("Speedup")
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw actual speedup
        chart
            .draw_series(LineSeries::new(
                speedups.iter().map(|(c, s)| (*c as f64, *s)),
                &BLUE,
            ))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?
            .label("Actual")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        // Draw markers
        chart
            .draw_series(
                speedups
                    .iter()
                    .map(|(c, s)| Circle::new((*c as f64, *s), 3, BLUE.filled())),
            )
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw ideal speedup line
        chart
            .draw_series(LineSeries::new(
                vec![(0.0, 0.0), (max_cores as f64, max_cores as f64)],
                &RED,
            ))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?
            .label("Ideal")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.9))
            .border_style(BLACK)
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }

    /// Plot efficiency curve (speedup / cores)
    pub fn efficiency_curve(
        &self,
        scaling_data: &[ScalingPoint],
        baseline: f64,
        path: &Path,
    ) -> VizResult<()> {
        if scaling_data.is_empty() {
            return Err(VizError::InvalidData("No scaling data".to_string()));
        }

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Compute efficiencies
        let efficiencies: Vec<(usize, f64)> = scaling_data
            .iter()
            .map(|p| (p.cores, (baseline / p.time) / p.cores as f64))
            .collect();

        let max_cores = efficiencies.iter().map(|(c, _)| *c).max().unwrap_or(1);

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..(max_cores as f64 * 1.1), 0.0..1.1)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .x_desc("Number of Cores")
            .y_desc("Efficiency")
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw efficiency curve
        chart
            .draw_series(LineSeries::new(
                efficiencies.iter().map(|(c, e)| (*c as f64, *e)),
                &BLUE,
            ))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .draw_series(
                efficiencies
                    .iter()
                    .map(|(c, e)| Circle::new((*c as f64, *e), 3, BLUE.filled())),
            )
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw ideal efficiency line (100%)
        chart
            .draw_series(LineSeries::new(
                vec![(0.0, 1.0), (max_cores as f64, 1.0)],
                &RED.mix(0.5),
            ))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }

    /// Plot time series performance data
    pub fn time_series(&self, times: &[(f64, f64)], path: &Path) -> VizResult<()> {
        if times.is_empty() {
            return Err(VizError::InvalidData("No time series data".to_string()));
        }

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        let x_min = times.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
        let x_max = times
            .iter()
            .map(|(x, _)| *x)
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = times.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
        let y_max = times
            .iter()
            .map(|(_, y)| *y)
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
            .x_desc(&self.config.x_axis.label)
            .y_desc(&self.config.y_axis.label)
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .draw_series(LineSeries::new(times.iter().copied(), &BLUE))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_plot_creation() {
        let config = PlotConfig::default();
        let _plot = PerfPlot::new(config);
    }

    #[test]
    fn test_empty_benchmarks() {
        let config = PlotConfig::default();
        let plot = PerfPlot::new(config);
        let results = vec![];
        let path = std::path::Path::new("/tmp/test.png");

        let result = plot.benchmark_comparison(&results, path);
        assert!(result.is_err());
    }

    #[test]
    fn test_speedup_computation() {
        let baseline = 10.0;
        let scaling_data = [
            ScalingPoint {
                cores: 1,
                time: 10.0,
            },
            ScalingPoint {
                cores: 2,
                time: 5.0,
            },
            ScalingPoint {
                cores: 4,
                time: 2.5,
            },
        ];

        // Speedup for 4 cores should be 10.0 / 2.5 = 4.0
        let speedup = baseline / scaling_data[2].time;
        assert!((speedup - 4.0).abs() < 1e-10);
    }
}
