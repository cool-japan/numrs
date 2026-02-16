//! 2D plotting module
//!
//! This module provides comprehensive 2D plotting capabilities including
//! line plots, scatter plots, bar charts, histograms, and area plots.
//!
//! # Example
//!
//! ```no_run
//! use numrs2::viz::{Plot2D, PlotConfig, Color};
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let x = Array1::linspace(0.0, 10.0, 100);
//! let y = x.mapv(|v: f64| v.sin());
//!
//! let mut plot = Plot2D::new(PlotConfig::default());
//! plot.line(&x, &y, "sin(x)")?;
//! plot.save("sine.png")?;
//! # Ok(())
//! # }
//! ```

use super::*;
use plotters::prelude::*;
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::path::Path;

// Re-import to avoid name collision with plotters::prelude::Color
use super::Color as VizColor;

/// 2D plot structure
pub struct Plot2D {
    config: PlotConfig,
    series: Vec<Series2D>,
}

/// A single data series in a 2D plot
#[derive(Clone)]
struct Series2D {
    x: Vec<f64>,
    y: Vec<f64>,
    label: String,
    style: SeriesStyle,
}

/// Style configuration for a data series
#[derive(Clone)]
pub struct SeriesStyle {
    /// Line color
    pub color: VizColor,
    /// Line style
    pub line_style: LineStyle,
    /// Line width
    pub line_width: LineWidth,
    /// Marker style
    pub marker_style: MarkerStyle,
    /// Marker size
    pub marker_size: u32,
}

impl Default for SeriesStyle {
    fn default() -> Self {
        Self {
            color: VizColor::BLUE,
            line_style: LineStyle::Solid,
            line_width: LineWidth::NORMAL,
            marker_style: MarkerStyle::None,
            marker_size: 3,
        }
    }
}

impl Plot2D {
    /// Create a new 2D plot with the given configuration
    pub fn new(config: PlotConfig) -> Self {
        Self {
            config,
            series: Vec::new(),
        }
    }

    /// Add a line plot
    pub fn line<S: AsRef<str>>(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        label: S,
    ) -> VizResult<&mut Self> {
        self.add_series(x, y, label, SeriesStyle::default())
    }

    /// Add a line plot with custom style
    pub fn line_styled<S: AsRef<str>>(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        label: S,
        style: SeriesStyle,
    ) -> VizResult<&mut Self> {
        self.add_series(x, y, label, style)
    }

    /// Add a scatter plot
    pub fn scatter<S: AsRef<str>>(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        label: S,
    ) -> VizResult<&mut Self> {
        let style = SeriesStyle {
            line_style: LineStyle::None,
            marker_style: MarkerStyle::Circle,
            marker_size: 5,
            ..Default::default()
        };
        self.add_series(x, y, label, style)
    }

    /// Add a scatter plot with custom style
    pub fn scatter_styled<S: AsRef<str>>(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        label: S,
        style: SeriesStyle,
    ) -> VizResult<&mut Self> {
        self.add_series(x, y, label, style)
    }

    /// Add a bar chart (vertical)
    pub fn bar<S: AsRef<str>>(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        label: S,
    ) -> VizResult<&mut Self> {
        if x.len() != y.len() {
            return Err(VizError::DimensionMismatch(format!(
                "x and y must have same length: {} != {}",
                x.len(),
                y.len()
            )));
        }

        self.series.push(Series2D {
            x: x.to_vec(),
            y: y.to_vec(),
            label: label.as_ref().to_string(),
            style: SeriesStyle::default(),
        });

        Ok(self)
    }

    /// Add a histogram
    pub fn histogram<S: AsRef<str>>(
        &mut self,
        data: &Array1<f64>,
        bins: usize,
        label: S,
    ) -> VizResult<&mut Self> {
        if bins == 0 {
            return Err(VizError::InvalidConfig(
                "Number of bins must be positive".to_string(),
            ));
        }

        let (hist, bin_edges) = compute_histogram(data, bins)?;

        // Convert histogram to bar chart format
        let x: Array1<f64> = Array1::from_vec(
            bin_edges
                .iter()
                .take(bin_edges.len() - 1)
                .zip(bin_edges.iter().skip(1))
                .map(|(a, b)| (a + b) / 2.0)
                .collect(),
        );

        self.bar(&x, &hist, label)
    }

    /// Add an area plot (filled region)
    pub fn area<S: AsRef<str>>(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        label: S,
    ) -> VizResult<&mut Self> {
        // For area plots, we'll use filled polygons in the save method
        self.add_series(x, y, label, SeriesStyle::default())
    }

    /// Add a generic series with custom style
    fn add_series<S: AsRef<str>>(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        label: S,
        style: SeriesStyle,
    ) -> VizResult<&mut Self> {
        if x.len() != y.len() {
            return Err(VizError::DimensionMismatch(format!(
                "x and y must have same length: {} != {}",
                x.len(),
                y.len()
            )));
        }

        if x.is_empty() {
            return Err(VizError::InvalidData(
                "Cannot plot empty arrays".to_string(),
            ));
        }

        self.series.push(Series2D {
            x: x.to_vec(),
            y: y.to_vec(),
            label: label.as_ref().to_string(),
            style,
        });

        Ok(self)
    }

    /// Save the plot to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> VizResult<()> {
        let path = path.as_ref();

        // Determine backend from file extension if not explicitly set
        let backend = match path.extension().and_then(|s| s.to_str()) {
            Some("png") => PlotBackend::Png,
            Some("svg") => PlotBackend::Svg,
            Some("html") => PlotBackend::Html,
            _ => self.config.backend,
        };

        match backend {
            PlotBackend::Png => self.save_png(path),
            PlotBackend::Svg => self.save_svg(path),
            PlotBackend::Html => self.save_html(path),
        }
    }

    /// Save as PNG
    fn save_png(&self, path: &Path) -> VizResult<()> {
        let root =
            BitMapBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        self.render_plot(root)?;
        Ok(())
    }

    /// Save as SVG
    fn save_svg(&self, path: &Path) -> VizResult<()> {
        let root =
            SVGBackend::new(path, (self.config.width, self.config.height)).into_drawing_area();

        self.render_plot(root)?;
        Ok(())
    }

    /// Save as HTML
    fn save_html(&self, path: &Path) -> VizResult<()> {
        // For HTML, we'll embed an SVG
        let svg_path = path.with_extension("svg");
        self.save_svg(&svg_path)?;

        // Read the SVG file
        let svg_content = std::fs::read_to_string(&svg_path).map_err(VizError::IoError)?;

        // Create HTML wrapper
        let html = format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ text-align: center; }}
    </style>
</head>
<body>
    <div class="plot-container">
        <h1>{}</h1>
        {}
    </div>
</body>
</html>"#,
            self.config.title, self.config.title, svg_content
        );

        std::fs::write(path, html)?;

        // Clean up temporary SVG
        let _ = std::fs::remove_file(&svg_path);

        Ok(())
    }

    /// Render the plot to a drawing area
    fn render_plot<DB: DrawingBackend>(
        &self,
        root: DrawingArea<DB, plotters::coord::Shift>,
    ) -> VizResult<()>
    where
        DB::ErrorType: 'static,
    {
        root.fill(&WHITE)
            .map_err(|e| VizError::RenderError(format!("Failed to fill background: {:?}", e)))?;

        // Compute data ranges
        let (x_min, x_max, y_min, y_max) = self.compute_ranges()?;

        // Apply axis range overrides if specified
        let x_min = self.config.x_axis.min.unwrap_or(x_min);
        let x_max = self.config.x_axis.max.unwrap_or(x_max);
        let y_min = self.config.y_axis.min.unwrap_or(y_min);
        let y_max = self.config.y_axis.max.unwrap_or(y_max);

        let mut chart = ChartBuilder::on(&root)
            .caption(&self.config.title, ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .map_err(|e| VizError::RenderError(format!("Failed to build chart: {:?}", e)))?;

        // Configure mesh (grid)
        let mut mesh = chart.configure_mesh();
        mesh.x_desc(&self.config.x_axis.label)
            .y_desc(&self.config.y_axis.label);

        if self.config.grid.show_major {
            mesh.draw()
                .map_err(|e| VizError::RenderError(format!("Failed to draw mesh: {:?}", e)))?;
        }

        // Draw each series
        for (idx, series) in self.series.iter().enumerate() {
            let color = self.get_series_color(&series.style.color, idx);
            let rgb = color.to_rgb_u8();
            let plot_color = RGBColor(rgb.0, rgb.1, rgb.2);

            // Draw line if not None
            if series.style.line_style != LineStyle::None {
                let line_series = LineSeries::new(
                    series.x.iter().zip(series.y.iter()).map(|(x, y)| (*x, *y)),
                    &plot_color,
                );

                chart
                    .draw_series(line_series)
                    .map_err(|e| VizError::RenderError(format!("Failed to draw series: {:?}", e)))?
                    .label(&series.label)
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], plot_color));
            }

            // Draw markers if not None
            if series.style.marker_style != MarkerStyle::None {
                let marker_series = series.x.iter().zip(series.y.iter()).map(|(x, y)| {
                    Circle::new(
                        (*x, *y),
                        series.style.marker_size as i32,
                        plot_color.filled(),
                    )
                });

                chart.draw_series(marker_series).map_err(|e| {
                    VizError::RenderError(format!("Failed to draw markers: {:?}", e))
                })?;
            }
        }

        // Configure legend if enabled
        if self.config.legend.show && !self.series.is_empty() {
            chart
                .configure_series_labels()
                .background_style(WHITE.mix(0.9))
                .border_style(BLACK)
                .draw()
                .map_err(|e| VizError::RenderError(format!("Failed to draw legend: {:?}", e)))?;
        }

        root.present()
            .map_err(|e| VizError::RenderError(format!("Failed to present plot: {:?}", e)))?;

        Ok(())
    }

    /// Compute data ranges for axes
    fn compute_ranges(&self) -> VizResult<(f64, f64, f64, f64)> {
        if self.series.is_empty() {
            return Err(VizError::InvalidData("No data to plot".to_string()));
        }

        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;

        for series in &self.series {
            for &x in &series.x {
                if x.is_finite() {
                    x_min = x_min.min(x);
                    x_max = x_max.max(x);
                }
            }
            for &y in &series.y {
                if y.is_finite() {
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                }
            }
        }

        if !x_min.is_finite() || !x_max.is_finite() || !y_min.is_finite() || !y_max.is_finite() {
            return Err(VizError::InvalidData(
                "All data points are non-finite".to_string(),
            ));
        }

        // Add 5% padding
        let x_range = x_max - x_min;
        let y_range = y_max - y_min;
        let x_padding = if x_range > 0.0 { x_range * 0.05 } else { 1.0 };
        let y_padding = if y_range > 0.0 { y_range * 0.05 } else { 1.0 };

        Ok((
            x_min - x_padding,
            x_max + x_padding,
            y_min - y_padding,
            y_max + y_padding,
        ))
    }

    /// Get color for a series, using default palette if needed
    fn get_series_color(&self, color: &VizColor, index: usize) -> VizColor {
        // If it's the default blue color, use palette
        if color == &VizColor::BLUE && index > 0 {
            DEFAULT_PALETTE[index % DEFAULT_PALETTE.len()]
        } else {
            *color
        }
    }
}

/// Compute histogram from data
fn compute_histogram(data: &Array1<f64>, bins: usize) -> VizResult<(Array1<f64>, Vec<f64>)> {
    if data.is_empty() {
        return Err(VizError::InvalidData(
            "Cannot compute histogram of empty data".to_string(),
        ));
    }

    // Find min and max
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for &val in data.iter() {
        if val.is_finite() {
            min = min.min(val);
            max = max.max(val);
        }
    }

    if !min.is_finite() || !max.is_finite() {
        return Err(VizError::InvalidData(
            "All data points are non-finite".to_string(),
        ));
    }

    // Create bin edges
    let bin_width = (max - min) / bins as f64;
    let bin_edges: Vec<f64> = (0..=bins).map(|i| min + i as f64 * bin_width).collect();

    // Count values in each bin
    let mut counts = vec![0.0; bins];
    for &val in data.iter() {
        if val.is_finite() {
            let bin_idx = ((val - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1); // Handle edge case where val == max
            counts[bin_idx] += 1.0;
        }
    }

    Ok((Array1::from_vec(counts), bin_edges))
}

/// Default color palette for multiple series
const DEFAULT_PALETTE: [VizColor; 10] = [
    VizColor::BLUE,
    VizColor::ORANGE,
    VizColor::GREEN,
    VizColor::RED,
    VizColor::PURPLE,
    VizColor::CYAN,
    VizColor::MAGENTA,
    VizColor::YELLOW,
    VizColor::GRAY,
    VizColor {
        r: 0.0,
        g: 0.5,
        b: 0.5,
        a: 1.0,
    }, // Teal
];

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_plot2d_creation() {
        let config = PlotConfig::default();
        let plot = Plot2D::new(config);
        assert_eq!(plot.series.len(), 0);
    }

    #[test]
    fn test_add_line() {
        let mut plot = Plot2D::new(PlotConfig::default());
        let x = Array1::linspace(0.0, 10.0, 11);
        let y = x.mapv(|v| v * 2.0);

        let result = plot.line(&x, &y, "test");
        assert!(result.is_ok());
        assert_eq!(plot.series.len(), 1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut plot = Plot2D::new(PlotConfig::default());
        let x = Array1::linspace(0.0, 10.0, 11);
        let y = Array1::linspace(0.0, 5.0, 6);

        let result = plot.line(&x, &y, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_histogram() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (hist, edges) = compute_histogram(&data, 5).expect("Histogram computation failed");

        assert_eq!(hist.len(), 5);
        assert_eq!(edges.len(), 6);
        assert_eq!(hist.sum(), 5.0);
    }

    #[test]
    fn test_empty_data() {
        let mut plot = Plot2D::new(PlotConfig::default());
        let x = Array1::from_vec(vec![]);
        let y = Array1::from_vec(vec![]);

        let result = plot.line(&x, &y, "test");
        assert!(result.is_err());
    }
}
