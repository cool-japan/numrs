//! 3D plotting module
//!
//! This module provides 3D visualization including surface plots,
//! contour plots, and 3D scatter plots.

use super::{PlotConfig, VizError, VizResult};
use plotters::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use std::path::Path;

// Import types to avoid confusion with plotters types
use super::ColorMap as VizColorMap;

/// 3D plot structure
pub struct Plot3D {
    config: PlotConfig,
}

impl Plot3D {
    /// Create a new 3D plot
    pub fn new(config: PlotConfig) -> Self {
        Self { config }
    }

    /// Create a surface plot from a 2D grid
    pub fn surface(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array2<f64>,
        path: &Path,
    ) -> VizResult<()> {
        if x.len() != z.ncols() || y.len() != z.nrows() {
            return Err(VizError::DimensionMismatch(format!(
                "Grid dimensions mismatch: x={}, y={}, z=({}, {})",
                x.len(),
                y.len(),
                z.nrows(),
                z.ncols()
            )));
        }

        super::ensure_fonts_registered();

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Find z range
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        for &val in z.iter() {
            if val.is_finite() {
                z_min = z_min.min(val);
                z_max = z_max.max(val);
            }
        }

        if !z_min.is_finite() || !z_max.is_finite() {
            return Err(VizError::InvalidData(
                "All z values are non-finite".to_string(),
            ));
        }

        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .build_cartesian_3d(x_min..x_max, z_min..z_max, y_min..y_max)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_axes()
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Draw surface (simplified - draw as points for now)
        let points: Vec<_> = x
            .iter()
            .enumerate()
            .flat_map(|(i, &x_val)| {
                y.iter().enumerate().map(move |(j, &y_val)| {
                    let z_val = z[[j, i]];
                    (x_val, z_val, y_val)
                })
            })
            .collect();

        chart
            .draw_series(PointSeries::of_element(
                points,
                3,
                &BLUE,
                &|coord, size, style| {
                    EmptyElement::at(coord) + Circle::new((0, 0), size, style.filled())
                },
            ))
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        root.present()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        Ok(())
    }

    /// Create a contour plot
    pub fn contour(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array2<f64>,
        path: &Path,
    ) -> VizResult<()> {
        if x.len() != z.ncols() || y.len() != z.nrows() {
            return Err(VizError::DimensionMismatch(format!(
                "Grid dimensions mismatch: x={}, y={}, z=({}, {})",
                x.len(),
                y.len(),
                z.nrows(),
                z.ncols()
            )));
        }

        super::ensure_fonts_registered();

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        // Find z range
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        for &val in z.iter() {
            if val.is_finite() {
                z_min = z_min.min(val);
                z_max = z_max.max(val);
            }
        }

        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

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

        // Draw contour as colored cells
        let z_range = z_max - z_min;
        for i in 0..y.len() {
            for j in 0..x.len() {
                let z_val = z[[i, j]];
                if z_val.is_finite() && z_range > 0.0 {
                    let normalized = (z_val - z_min) / z_range;
                    let color = VizColorMap::Viridis.get_color(normalized);
                    let rgb = color.to_rgb_u8();
                    let plot_color = RGBColor(rgb.0, rgb.1, rgb.2);

                    let x0 = x[j];
                    let x1 = if j + 1 < x.len() {
                        x[j + 1]
                    } else {
                        x[j] + (x[j] - x[j - 1])
                    };
                    let y0 = y[i];
                    let y1 = if i + 1 < y.len() {
                        y[i + 1]
                    } else {
                        y[i] + (y[i] - y[i - 1])
                    };

                    chart
                        .draw_series(std::iter::once(Rectangle::new(
                            [(x0, y0), (x1, y1)],
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

    /// Create a 3D scatter plot
    pub fn scatter3d(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array1<f64>,
        path: &Path,
    ) -> VizResult<()> {
        if x.len() != y.len() || x.len() != z.len() {
            return Err(VizError::DimensionMismatch(format!(
                "Array lengths must match: x={}, y={}, z={}",
                x.len(),
                y.len(),
                z.len()
            )));
        }

        super::ensure_fonts_registered();

        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let z_min = z.iter().cloned().fold(f64::INFINITY, f64::min);
        let z_max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .build_cartesian_3d(x_min..x_max, z_min..z_max, y_min..y_max)
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        chart
            .configure_axes()
            .draw()
            .map_err(|e| VizError::RenderError(format!("{:?}", e)))?;

        let points: Vec<_> = x
            .iter()
            .zip(y.iter())
            .zip(z.iter())
            .map(|((&x, &y), &z)| (x, z, y))
            .collect();

        chart
            .draw_series(PointSeries::of_element(
                points,
                5,
                &BLUE,
                &|coord, size, style| {
                    EmptyElement::at(coord) + Circle::new((0, 0), size, style.filled())
                },
            ))
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
    fn test_plot3d_creation() {
        let config = PlotConfig::default();
        let _plot = Plot3D::new(config);
    }

    #[test]
    fn test_scatter3d_dimension_mismatch() {
        let config = PlotConfig::default();
        let plot = Plot3D::new(config);

        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);
        let z = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let path = std::path::Path::new("/tmp/test.png");
        let result = plot.scatter3d(&x, &y, &z, path);
        assert!(result.is_err());
    }
}
