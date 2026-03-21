//! Matrix and tensor visualization module
//!
//! This module provides visualization for matrices including heatmaps,
//! spy plots for sparse matrices, and eigenvalue visualizations.

use super::*;
use plotters::prelude::*;
use plotters::style::Color;
use scirs2_core::ndarray::{Array1, Array2};
use std::path::Path;

// Import types to avoid confusion with plotters types
use super::ColorMap as VizColorMap;

/// Matrix plot structure
pub struct MatrixPlot {
    config: PlotConfig,
    colormap: VizColorMap,
}

impl MatrixPlot {
    /// Create a new matrix plot
    pub fn new(config: PlotConfig) -> Self {
        Self {
            config,
            colormap: VizColorMap::Viridis,
        }
    }

    /// Set the color map
    pub fn with_colormap(mut self, colormap: VizColorMap) -> Self {
        self.colormap = colormap;
        self
    }

    /// Create a heatmap from a 2D array
    pub fn heatmap(&self, data: &Array2<f64>, path: &Path) -> VizResult<()> {
        let (rows, cols) = data.dim();
        if rows == 0 || cols == 0 {
            return Err(VizError::InvalidData("Empty matrix".to_string()));
        }

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Find min/max for normalization
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for &val in data.iter() {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        if !min_val.is_finite() || !max_val.is_finite() {
            return Err(VizError::InvalidData(
                "All values are non-finite".to_string(),
            ));
        }

        let range = max_val - min_val;
        if range == 0.0 {
            return Err(VizError::InvalidData(
                "All values are identical".to_string(),
            ));
        }

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0..cols as i32, 0..rows as i32)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw heatmap cells
        for i in 0..rows {
            for j in 0..cols {
                let val = data[[i, j]];
                if val.is_finite() {
                    let normalized = (val - min_val) / range;
                    let color = self.colormap.get_color(normalized);
                    let rgb = color.to_rgb_u8();
                    let plot_color = RGBColor(rgb.0, rgb.1, rgb.2);

                    chart
                        .draw_series(std::iter::once(Rectangle::new(
                            [(j as i32, i as i32), ((j + 1) as i32, (i + 1) as i32)],
                            plot_color.filled(),
                        )))
                        .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;
                }
            }
        }

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }

    /// Create a spy plot showing the sparsity pattern
    pub fn spy(&self, data: &Array2<f64>, path: &Path) -> VizResult<()> {
        let (rows, cols) = data.dim();
        if rows == 0 || cols == 0 {
            return Err(VizError::InvalidData("Empty matrix".to_string()));
        }

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0..cols as i32, 0..rows as i32)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .disable_x_mesh()
            .disable_y_mesh()
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw non-zero elements
        for i in 0..rows {
            for j in 0..cols {
                if data[[i, j]].abs() > 1e-10 {
                    chart
                        .draw_series(std::iter::once(Circle::new(
                            (j as i32, i as i32),
                            2,
                            BLACK.filled(),
                        )))
                        .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;
                }
            }
        }

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }

    /// Plot eigenvalues
    pub fn eigenvalues(&self, eigenvalues: &Array1<f64>, path: &Path) -> VizResult<()> {
        if eigenvalues.is_empty() {
            return Err(VizError::InvalidData("No eigenvalues".to_string()));
        }

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        let indices: Vec<f64> = (0..eigenvalues.len()).map(|i| i as f64).collect();
        let min_ev = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ev = eigenvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0.0..(eigenvalues.len() as f64), min_ev..max_ev)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_mesh()
            .x_desc("Index")
            .y_desc("Eigenvalue")
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Plot eigenvalues
        chart
            .draw_series(LineSeries::new(
                indices
                    .iter()
                    .zip(eigenvalues.iter())
                    .map(|(&i, &v)| (i, v)),
                &BLUE,
            ))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .draw_series(
                indices
                    .iter()
                    .zip(eigenvalues.iter())
                    .map(|(&i, &v)| Circle::new((i, v), 3, RED.filled())),
            )
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_matrix_plot_creation() {
        let config = PlotConfig::default();
        let _plot = MatrixPlot::new(config);
    }

    #[test]
    fn test_empty_matrix() {
        let config = PlotConfig::default();
        let plot = MatrixPlot::new(config);
        let empty = Array2::<f64>::zeros((0, 0));
        let path = std::path::Path::new("/tmp/test.png");

        let result = plot.heatmap(&empty, path);
        assert!(result.is_err());
    }
}
