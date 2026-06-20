//! Export format module
//!
//! This module provides utilities for exporting plots to various formats
//! including PNG, SVG, HTML, and LaTeX/TikZ.

use super::{VizError, VizResult};
use std::path::Path;

/// Export format enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// PNG bitmap format
    Png,
    /// SVG vector format
    Svg,
    /// HTML with embedded visualization
    Html,
    /// LaTeX TikZ format
    Tikz,
}

impl ExportFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> VizResult<Self> {
        match path.extension().and_then(|s| s.to_str()) {
            Some("png") => Ok(ExportFormat::Png),
            Some("svg") => Ok(ExportFormat::Svg),
            Some("html") | Some("htm") => Ok(ExportFormat::Html),
            Some("tex") | Some("tikz") => Ok(ExportFormat::Tikz),
            Some(ext) => Err(VizError::ExportError(format!(
                "Unsupported format: {}",
                ext
            ))),
            None => Err(VizError::ExportError("No file extension found".to_string())),
        }
    }

    /// Get the default extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            ExportFormat::Png => "png",
            ExportFormat::Svg => "svg",
            ExportFormat::Html => "html",
            ExportFormat::Tikz => "tex",
        }
    }
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Output format
    pub format: ExportFormat,
    /// Image width in pixels (for bitmap formats)
    pub width: u32,
    /// Image height in pixels (for bitmap formats)
    pub height: u32,
    /// DPI for bitmap formats
    pub dpi: u32,
    /// Quality factor (0-100) for lossy formats
    pub quality: u8,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Png,
            width: 800,
            height: 600,
            dpi: 96,
            quality: 90,
        }
    }
}

impl ExportConfig {
    /// Create a new export configuration with the specified format
    pub fn new(format: ExportFormat) -> Self {
        Self {
            format,
            ..Default::default()
        }
    }

    /// Set the output dimensions
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

    /// Set the DPI
    pub fn with_dpi(mut self, dpi: u32) -> VizResult<Self> {
        if dpi == 0 {
            return Err(VizError::InvalidConfig("DPI must be positive".to_string()));
        }
        self.dpi = dpi;
        Ok(self)
    }

    /// Set the quality
    pub fn with_quality(mut self, quality: u8) -> VizResult<Self> {
        if quality > 100 {
            return Err(VizError::InvalidConfig(
                "Quality must be between 0 and 100".to_string(),
            ));
        }
        self.quality = quality;
        Ok(self)
    }
}

/// Export helper functions
pub struct Exporter;

impl Exporter {
    /// Export plot data to TikZ format
    pub fn to_tikz(
        x: &[f64],
        y: &[f64],
        title: &str,
        xlabel: &str,
        ylabel: &str,
        path: &Path,
    ) -> VizResult<()> {
        if x.len() != y.len() {
            return Err(VizError::DimensionMismatch(format!(
                "x and y must have same length: {} != {}",
                x.len(),
                y.len()
            )));
        }

        let mut tikz = String::new();

        tikz.push_str("\\begin{tikzpicture}\n");
        tikz.push_str("\\begin{axis}[\n");
        tikz.push_str(&format!("    title={{{}}},\n", title));
        tikz.push_str(&format!("    xlabel={{{}}},\n", xlabel));
        tikz.push_str(&format!("    ylabel={{{}}},\n", ylabel));
        tikz.push_str("    grid=major,\n");
        tikz.push_str("    legend pos=north west,\n");
        tikz.push_str("]\n");
        tikz.push_str("\\addplot coordinates {\n");

        for (xi, yi) in x.iter().zip(y.iter()) {
            tikz.push_str(&format!("    ({}, {})\n", xi, yi));
        }

        tikz.push_str("};\n");
        tikz.push_str("\\end{axis}\n");
        tikz.push_str("\\end{tikzpicture}\n");

        std::fs::write(path, tikz)?;

        Ok(())
    }

    /// Create a standalone LaTeX document with TikZ plot
    pub fn to_tikz_standalone(
        x: &[f64],
        y: &[f64],
        title: &str,
        xlabel: &str,
        ylabel: &str,
        path: &Path,
    ) -> VizResult<()> {
        if x.len() != y.len() {
            return Err(VizError::DimensionMismatch(format!(
                "x and y must have same length: {} != {}",
                x.len(),
                y.len()
            )));
        }

        let mut latex = String::new();

        latex.push_str("\\documentclass{standalone}\n");
        latex.push_str("\\usepackage{pgfplots}\n");
        latex.push_str("\\pgfplotsset{compat=1.18}\n");
        latex.push_str("\\begin{document}\n");

        latex.push_str("\\begin{tikzpicture}\n");
        latex.push_str("\\begin{axis}[\n");
        latex.push_str(&format!("    title={{{}}},\n", title));
        latex.push_str(&format!("    xlabel={{{}}},\n", xlabel));
        latex.push_str(&format!("    ylabel={{{}}},\n", ylabel));
        latex.push_str("    grid=major,\n");
        latex.push_str("    legend pos=north west,\n");
        latex.push_str("]\n");
        latex.push_str("\\addplot coordinates {\n");

        for (xi, yi) in x.iter().zip(y.iter()) {
            latex.push_str(&format!("    ({}, {})\n", xi, yi));
        }

        latex.push_str("};\n");
        latex.push_str("\\end{axis}\n");
        latex.push_str("\\end{tikzpicture}\n");

        latex.push_str("\\end{document}\n");

        std::fs::write(path, latex)?;

        Ok(())
    }

    /// Batch export to multiple formats
    pub fn batch_export<F>(
        base_path: &Path,
        formats: &[ExportFormat],
        plot_fn: F,
    ) -> VizResult<Vec<std::path::PathBuf>>
    where
        F: Fn(&Path) -> VizResult<()>,
    {
        let mut output_paths = Vec::new();

        for format in formats {
            let path = base_path.with_extension(format.extension());
            plot_fn(&path)?;
            output_paths.push(path);
        }

        Ok(output_paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_format_from_extension() {
        let path = PathBuf::from("test.png");
        let format = ExportFormat::from_extension(&path);
        assert!(format.is_ok());
        assert_eq!(format.unwrap_or(ExportFormat::Svg), ExportFormat::Png);

        let path = PathBuf::from("test.svg");
        let format = ExportFormat::from_extension(&path);
        assert_eq!(format.unwrap_or(ExportFormat::Png), ExportFormat::Svg);

        let path = PathBuf::from("test.unknown");
        let format = ExportFormat::from_extension(&path);
        assert!(format.is_err());
    }

    #[test]
    fn test_export_config() {
        let config = ExportConfig::new(ExportFormat::Png)
            .with_size(1024, 768)
            .expect("Failed to set size")
            .with_dpi(150)
            .expect("Failed to set DPI")
            .with_quality(95)
            .expect("Failed to set quality");

        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
        assert_eq!(config.dpi, 150);
        assert_eq!(config.quality, 95);
    }

    #[test]
    fn test_invalid_config() {
        let result = ExportConfig::new(ExportFormat::Png).with_size(0, 600);
        assert!(result.is_err());

        let result = ExportConfig::new(ExportFormat::Png).with_quality(150);
        assert!(result.is_err());
    }

    #[test]
    fn test_tikz_dimension_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        let path = PathBuf::from("/tmp/test.tex");

        let result = Exporter::to_tikz(&x, &y, "Test", "X", "Y", &path);
        assert!(result.is_err());
    }
}
