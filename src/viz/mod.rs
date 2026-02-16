//! Advanced visualization tools for NumRS2
//!
//! This module provides comprehensive plotting and visualization capabilities
//! for numerical data, matrices, and statistical analysis.
//!
//! # Features
//!
//! - 2D plotting (line, scatter, bar, histogram, area)
//! - Statistical plots (box, violin, Q-Q, heatmap)
//! - Matrix/tensor visualization (heatmap, spy plots)
//! - 3D visualization (surface, contour, scatter)
//! - Performance visualization (benchmarks, scaling analysis)
//! - Multiple export formats (PNG, SVG, HTML, LaTeX/TikZ)
//!
//! # Example
//!
//! ```no_run
//! use numrs2::viz::{Plot2D, PlotConfig};
//! use scirs2_core::ndarray::Array1;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let x = Array1::linspace(0.0, 10.0, 100);
//! let y = x.mapv(|val: f64| val.sin());
//!
//! let mut plot = Plot2D::new(PlotConfig::default());
//! plot.line(&x, &y, "sin(x)")?;
//! plot.save("output.png")?;
//! # Ok(())
//! # }
//! ```

// Re-export submodules
pub mod export;
pub mod matrix;
pub mod perf;
pub mod plot2d;
pub mod plot3d;
pub mod stats;

// Re-export commonly used types
pub use export::{ExportConfig, ExportFormat, Exporter};
pub use matrix::MatrixPlot;
pub use perf::{BenchmarkResult, PerfPlot, ScalingPoint};
pub use plot2d::Plot2D;
pub use plot3d::Plot3D;
pub use stats::{BinStrategy, StatPlot};

use thiserror::Error;

/// Visualization error types
#[derive(Error, Debug)]
pub enum VizError {
    /// Error during plot rendering
    #[error("Plot rendering error: {0}")]
    RenderError(String),

    /// Error during file I/O
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid data provided
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Backend error
    #[error("Backend error: {0}")]
    BackendError(String),

    /// Export format error
    #[error("Export error: {0}")]
    ExportError(String),

    /// Dimension mismatch
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

/// Result type for visualization operations
pub type VizResult<T> = Result<T, VizError>;

/// RGB color representation
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    /// Red component (0.0 - 1.0)
    pub r: f64,
    /// Green component (0.0 - 1.0)
    pub g: f64,
    /// Blue component (0.0 - 1.0)
    pub b: f64,
    /// Alpha component (0.0 - 1.0)
    pub a: f64,
}

impl Color {
    /// Create a new color from RGB components
    pub fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Create a new color from RGBA components
    pub fn rgba(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    /// Create color from hex string (e.g., "#FF0000" or "#FF0000FF")
    pub fn from_hex(hex: &str) -> VizResult<Self> {
        let hex = hex.trim_start_matches('#');

        let (r, g, b, a) = match hex.len() {
            6 => {
                let r = u8::from_str_radix(&hex[0..2], 16)
                    .map_err(|_| VizError::InvalidData("Invalid hex color".to_string()))?;
                let g = u8::from_str_radix(&hex[2..4], 16)
                    .map_err(|_| VizError::InvalidData("Invalid hex color".to_string()))?;
                let b = u8::from_str_radix(&hex[4..6], 16)
                    .map_err(|_| VizError::InvalidData("Invalid hex color".to_string()))?;
                (r, g, b, 255)
            }
            8 => {
                let r = u8::from_str_radix(&hex[0..2], 16)
                    .map_err(|_| VizError::InvalidData("Invalid hex color".to_string()))?;
                let g = u8::from_str_radix(&hex[2..4], 16)
                    .map_err(|_| VizError::InvalidData("Invalid hex color".to_string()))?;
                let b = u8::from_str_radix(&hex[4..6], 16)
                    .map_err(|_| VizError::InvalidData("Invalid hex color".to_string()))?;
                let a = u8::from_str_radix(&hex[6..8], 16)
                    .map_err(|_| VizError::InvalidData("Invalid hex color".to_string()))?;
                (r, g, b, a)
            }
            _ => {
                return Err(VizError::InvalidData(
                    "Hex color must be 6 or 8 characters".to_string(),
                ))
            }
        };

        Ok(Self {
            r: r as f64 / 255.0,
            g: g as f64 / 255.0,
            b: b as f64 / 255.0,
            a: a as f64 / 255.0,
        })
    }

    /// Convert to RGB tuple (u8 values)
    pub fn to_rgb_u8(&self) -> (u8, u8, u8) {
        (
            (self.r * 255.0) as u8,
            (self.g * 255.0) as u8,
            (self.b * 255.0) as u8,
        )
    }

    /// Convert to RGBA tuple (u8 values)
    pub fn to_rgba_u8(&self) -> (u8, u8, u8, u8) {
        (
            (self.r * 255.0) as u8,
            (self.g * 255.0) as u8,
            (self.b * 255.0) as u8,
            (self.a * 255.0) as u8,
        )
    }

    // Predefined colors
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const RED: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const GREEN: Color = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    pub const BLUE: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    pub const YELLOW: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    pub const CYAN: Color = Color {
        r: 0.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const MAGENTA: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    pub const ORANGE: Color = Color {
        r: 1.0,
        g: 0.647,
        b: 0.0,
        a: 1.0,
    };
    pub const PURPLE: Color = Color {
        r: 0.5,
        g: 0.0,
        b: 0.5,
        a: 1.0,
    };
    pub const GRAY: Color = Color {
        r: 0.5,
        g: 0.5,
        b: 0.5,
        a: 1.0,
    };
}

/// Line style for plotting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineStyle {
    /// Solid line
    Solid,
    /// Dashed line
    Dashed,
    /// Dotted line
    Dotted,
    /// Dash-dot line
    DashDot,
    /// No line (only markers)
    None,
}

/// Marker style for scatter plots
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerStyle {
    /// Circle marker
    Circle,
    /// Square marker
    Square,
    /// Triangle marker
    Triangle,
    /// Diamond marker
    Diamond,
    /// Plus (+) marker
    Plus,
    /// Cross (x) marker
    Cross,
    /// Star marker
    Star,
    /// No marker (only line)
    None,
}

/// Line width specification
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineWidth(pub f64);

impl LineWidth {
    /// Create a new line width
    pub fn new(width: f64) -> VizResult<Self> {
        if width > 0.0 {
            Ok(Self(width))
        } else {
            Err(VizError::InvalidConfig(
                "Line width must be positive".to_string(),
            ))
        }
    }

    /// Thin line (1.0)
    pub const THIN: LineWidth = LineWidth(1.0);

    /// Normal line (2.0)
    pub const NORMAL: LineWidth = LineWidth(2.0);

    /// Thick line (3.0)
    pub const THICK: LineWidth = LineWidth(3.0);

    /// Very thick line (5.0)
    pub const VERY_THICK: LineWidth = LineWidth(5.0);
}

impl Default for LineWidth {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// Plot backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotBackend {
    /// PNG bitmap output
    Png,
    /// SVG vector output
    Svg,
    /// HTML with embedded SVG
    Html,
}

/// Grid configuration
#[derive(Debug, Clone)]
pub struct GridConfig {
    /// Show major grid lines
    pub show_major: bool,
    /// Show minor grid lines
    pub show_minor: bool,
    /// Major grid color
    pub major_color: Color,
    /// Minor grid color
    pub minor_color: Color,
    /// Major grid line style
    pub major_style: LineStyle,
    /// Minor grid line style
    pub minor_style: LineStyle,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            show_major: true,
            show_minor: false,
            major_color: Color::rgba(0.8, 0.8, 0.8, 0.5),
            minor_color: Color::rgba(0.9, 0.9, 0.9, 0.3),
            major_style: LineStyle::Solid,
            minor_style: LineStyle::Dotted,
        }
    }
}

/// Axis configuration
#[derive(Debug, Clone)]
pub struct AxisConfig {
    /// Axis label
    pub label: String,
    /// Minimum value (auto if None)
    pub min: Option<f64>,
    /// Maximum value (auto if None)
    pub max: Option<f64>,
    /// Use logarithmic scale
    pub log_scale: bool,
    /// Show axis
    pub show: bool,
}

impl Default for AxisConfig {
    fn default() -> Self {
        Self {
            label: String::new(),
            min: None,
            max: None,
            log_scale: false,
            show: true,
        }
    }
}

impl AxisConfig {
    /// Create a new axis configuration with a label
    pub fn with_label(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Default::default()
        }
    }

    /// Set the axis range
    pub fn with_range(mut self, min: f64, max: f64) -> VizResult<Self> {
        if max <= min {
            return Err(VizError::InvalidConfig(
                "Max must be greater than min".to_string(),
            ));
        }
        self.min = Some(min);
        self.max = Some(max);
        Ok(self)
    }

    /// Enable logarithmic scale
    pub fn with_log_scale(mut self) -> Self {
        self.log_scale = true;
        self
    }
}

/// Legend configuration
#[derive(Debug, Clone)]
pub struct LegendConfig {
    /// Show legend
    pub show: bool,
    /// Legend position
    pub position: LegendPosition,
    /// Background color
    pub background: Color,
    /// Border color
    pub border_color: Color,
}

impl Default for LegendConfig {
    fn default() -> Self {
        Self {
            show: true,
            position: LegendPosition::TopRight,
            background: Color::rgba(1.0, 1.0, 1.0, 0.9),
            border_color: Color::BLACK,
        }
    }
}

/// Legend position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegendPosition {
    /// Top left corner
    TopLeft,
    /// Top right corner
    TopRight,
    /// Bottom left corner
    BottomLeft,
    /// Bottom right corner
    BottomRight,
    /// Outside right
    OutsideRight,
    /// Outside bottom
    OutsideBottom,
}

/// Plot configuration
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Plot title
    pub title: String,
    /// X-axis configuration
    pub x_axis: AxisConfig,
    /// Y-axis configuration
    pub y_axis: AxisConfig,
    /// Grid configuration
    pub grid: GridConfig,
    /// Legend configuration
    pub legend: LegendConfig,
    /// Plot width in pixels
    pub width: u32,
    /// Plot height in pixels
    pub height: u32,
    /// Background color
    pub background: Color,
    /// Plot backend
    pub backend: PlotBackend,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            title: String::new(),
            x_axis: AxisConfig::default(),
            y_axis: AxisConfig::default(),
            grid: GridConfig::default(),
            legend: LegendConfig::default(),
            width: 800,
            height: 600,
            background: Color::WHITE,
            backend: PlotBackend::Png,
        }
    }
}

impl PlotConfig {
    /// Create a new plot configuration with a title
    pub fn with_title(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }

    /// Set plot dimensions
    pub fn with_size(mut self, width: u32, height: u32) -> VizResult<Self> {
        if width == 0 || height == 0 {
            return Err(VizError::InvalidConfig(
                "Width and height must be positive".to_string(),
            ));
        }
        self.width = width;
        self.height = height;
        Ok(self)
    }

    /// Set the backend
    pub fn with_backend(mut self, backend: PlotBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set x-axis label
    pub fn with_x_label(mut self, label: impl Into<String>) -> Self {
        self.x_axis.label = label.into();
        self
    }

    /// Set y-axis label
    pub fn with_y_label(mut self, label: impl Into<String>) -> Self {
        self.y_axis.label = label.into();
        self
    }
}

/// Color map for heatmaps and matrix visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMap {
    /// Viridis color map (perceptually uniform)
    Viridis,
    /// Plasma color map
    Plasma,
    /// Inferno color map
    Inferno,
    /// Magma color map
    Magma,
    /// Cool-warm diverging color map
    Coolwarm,
    /// Red-blue diverging color map
    RdBu,
    /// Grayscale
    Gray,
    /// Hot (black-red-yellow-white)
    Hot,
    /// Jet (rainbow)
    Jet,
}

impl ColorMap {
    /// Get color from the color map for a value in [0, 1]
    pub fn get_color(&self, value: f64) -> Color {
        let value = value.clamp(0.0, 1.0);

        match self {
            ColorMap::Viridis => viridis(value),
            ColorMap::Plasma => plasma(value),
            ColorMap::Inferno => inferno(value),
            ColorMap::Magma => magma(value),
            ColorMap::Coolwarm => coolwarm(value),
            ColorMap::RdBu => rdbu(value),
            ColorMap::Gray => Color::rgb(value, value, value),
            ColorMap::Hot => hot(value),
            ColorMap::Jet => jet(value),
        }
    }
}

/// Viridis color map
fn viridis(t: f64) -> Color {
    // Simplified viridis approximation
    let r = 0.267 + 0.004 * t + 0.328 * t * t;
    let g = 0.005 + 0.506 * t + 0.344 * t * t;
    let b = 0.329 + 1.112 * t - 0.722 * t * t;
    Color::rgb(r, g, b)
}

/// Plasma color map
fn plasma(t: f64) -> Color {
    let r = 0.050 + 1.475 * t - 0.823 * t * t;
    let g = 0.029 + 0.234 * t + 2.091 * t * t - 2.174 * t * t * t;
    let b = 0.533 + 1.206 * t - 2.614 * t * t + 1.909 * t * t * t;
    Color::rgb(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

/// Inferno color map
fn inferno(t: f64) -> Color {
    let r = -0.002 + 1.815 * t - 1.259 * t * t + 0.699 * t * t * t;
    let g = 0.001 + 0.209 * t + 1.398 * t * t - 0.908 * t * t * t;
    let b = 0.015 + 2.955 * t - 7.677 * t * t + 5.172 * t * t * t;
    Color::rgb(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

/// Magma color map
fn magma(t: f64) -> Color {
    let r = -0.002 + 1.797 * t - 1.224 * t * t + 0.663 * t * t * t;
    let g = 0.001 + 0.226 * t + 1.368 * t * t - 0.896 * t * t * t;
    let b = 0.332 + 2.293 * t - 5.538 * t * t + 3.371 * t * t * t;
    Color::rgb(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

/// Cool-warm diverging color map
fn coolwarm(t: f64) -> Color {
    let r = 0.231 + 2.113 * t - 2.420 * t * t + 1.076 * t * t * t;
    let g = 0.299 + 1.205 * t - 1.066 * t * t;
    let b = 0.754 - 1.021 * t + 0.267 * t * t;
    Color::rgb(r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

/// Red-blue diverging color map
fn rdbu(t: f64) -> Color {
    if t < 0.5 {
        let s = t * 2.0;
        Color::rgb(s, s, 1.0)
    } else {
        let s = (t - 0.5) * 2.0;
        Color::rgb(1.0, 1.0 - s, 1.0 - s)
    }
}

/// Hot color map (black-red-yellow-white)
fn hot(t: f64) -> Color {
    let r = (t * 3.0).clamp(0.0, 1.0);
    let g = ((t - 0.333) * 3.0).clamp(0.0, 1.0);
    let b = ((t - 0.666) * 3.0).clamp(0.0, 1.0);
    Color::rgb(r, g, b)
}

/// Jet color map (rainbow)
fn jet(t: f64) -> Color {
    let r = ((t - 0.5) * 4.0).clamp(0.0, 1.0);
    let g = (1.0 - (t - 0.5).abs() * 4.0).clamp(0.0, 1.0);
    let b = ((0.5 - t) * 4.0).clamp(0.0, 1.0);
    Color::rgb(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex() {
        let color = Color::from_hex("#FF0000").expect("Failed to parse hex color");
        assert_eq!(color.r, 1.0);
        assert_eq!(color.g, 0.0);
        assert_eq!(color.b, 0.0);
        assert_eq!(color.a, 1.0);

        let color = Color::from_hex("#FF000080").expect("Failed to parse hex color with alpha");
        assert!((color.a - 0.5019607843137255).abs() < 1e-6);
    }

    #[test]
    fn test_color_to_rgb() {
        let color = Color::RED;
        let (r, g, b) = color.to_rgb_u8();
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_line_width() {
        let width = LineWidth::new(2.5).expect("Failed to create line width");
        assert_eq!(width.0, 2.5);

        assert!(LineWidth::new(-1.0).is_err());
        assert!(LineWidth::new(0.0).is_err());
    }

    #[test]
    fn test_axis_config() {
        let axis = AxisConfig::with_label("X Axis");
        assert_eq!(axis.label, "X Axis");

        let axis = axis.with_range(0.0, 10.0).expect("Failed to set range");
        assert_eq!(axis.min, Some(0.0));
        assert_eq!(axis.max, Some(10.0));

        assert!(AxisConfig::default().with_range(10.0, 5.0).is_err());
    }

    #[test]
    fn test_plot_config() {
        let config = PlotConfig::with_title("Test Plot")
            .with_x_label("X")
            .with_y_label("Y");

        assert_eq!(config.title, "Test Plot");
        assert_eq!(config.x_axis.label, "X");
        assert_eq!(config.y_axis.label, "Y");

        let config = config.with_size(1024, 768).expect("Failed to set size");
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);

        assert!(PlotConfig::default().with_size(0, 600).is_err());
    }

    #[test]
    fn test_color_map() {
        let color = ColorMap::Viridis.get_color(0.5);
        assert!(color.r >= 0.0 && color.r <= 1.0);
        assert!(color.g >= 0.0 && color.g <= 1.0);
        assert!(color.b >= 0.0 && color.b <= 1.0);

        // Test clamping
        let color = ColorMap::Gray.get_color(-0.5);
        assert_eq!(color.r, 0.0);

        let color = ColorMap::Gray.get_color(1.5);
        assert_eq!(color.r, 1.0);
    }
}
