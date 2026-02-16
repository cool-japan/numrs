//! Comprehensive tests for NumRS2 visualization module

#![cfg(feature = "visualization")]

use numrs2::viz::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Normal};
use std::path::PathBuf;
use tempfile::TempDir;

/// Helper function to create a temporary directory for tests
fn setup_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temp dir")
}

#[test]
fn test_plot2d_line() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("line.png");

    let x = Array1::linspace(0.0, 10.0, 100);
    let y = x.mapv(|v: f64| v.sin());

    let config = PlotConfig::with_title("Test Line Plot");
    let mut plot = Plot2D::new(config);

    assert!(plot.line(&x, &y, "sin(x)").is_ok());
    assert!(plot.save(&path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_plot2d_scatter() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("scatter.png");

    let x = Array1::from_vec((0..50).map(|i| i as f64).collect());
    let y = x.mapv(|v| v * v);

    let config = PlotConfig::default();
    let mut plot = Plot2D::new(config);

    assert!(plot.scatter(&x, &y, "data").is_ok());
    assert!(plot.save(&path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_plot2d_multiple_series() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("multi.png");

    let x = Array1::linspace(0.0, 10.0, 50);
    let y1 = x.mapv(|v: f64| v.sin());
    let y2 = x.mapv(|v: f64| v.cos());

    let config = PlotConfig::with_title("Multiple Series");
    let mut plot = Plot2D::new(config);

    assert!(plot.line(&x, &y1, "sin").is_ok());
    assert!(plot.line(&x, &y2, "cos").is_ok());
    assert!(plot.save(&path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_plot2d_empty_data() {
    let config = PlotConfig::default();
    let mut plot = Plot2D::new(config);

    let x = Array1::from_vec(vec![]);
    let y = Array1::from_vec(vec![]);

    assert!(plot.line(&x, &y, "empty").is_err());
}

#[test]
fn test_plot2d_dimension_mismatch() {
    let config = PlotConfig::default();
    let mut plot = Plot2D::new(config);

    let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let y = Array1::from_vec(vec![1.0, 2.0]);

    assert!(plot.line(&x, &y, "mismatch").is_err());
}

#[test]
fn test_histogram() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("hist.png");

    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).expect("Failed to create distribution");
    let data: Vec<f64> = (0..1000).map(|_| normal.sample(&mut rng)).collect();
    let data_arr = Array1::from_vec(data);

    let config = PlotConfig::with_title("Histogram Test");
    let stat_plot = StatPlot::new(config);

    assert!(stat_plot
        .histogram(&data_arr, BinStrategy::Sturges, &path)
        .is_ok());
    assert!(path.exists());
}

#[test]
fn test_boxplot() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("box.png");

    let data = Array1::from_vec((1..=100).map(|x| x as f64).collect());

    let config = PlotConfig::with_title("Box Plot Test");
    let stat_plot = StatPlot::new(config);

    assert!(stat_plot.boxplot(&data, &path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_qqplot() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("qq.png");

    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).expect("Failed to create distribution");
    let data: Vec<f64> = (0..100).map(|_| normal.sample(&mut rng)).collect();
    let data_arr = Array1::from_vec(data);

    let config = PlotConfig::with_title("Q-Q Plot Test");
    let stat_plot = StatPlot::new(config);

    assert!(stat_plot.qqplot(&data_arr, &path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_heatmap() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("heatmap.png");

    let mut data = Array2::zeros((10, 10));
    for i in 0..10 {
        for j in 0..10 {
            data[[i, j]] = (i + j) as f64;
        }
    }

    let config = PlotConfig::with_title("Heatmap Test");
    let matrix_plot = MatrixPlot::new(config);

    assert!(matrix_plot.heatmap(&data, &path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_spy_plot() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("spy.png");

    let mut data = Array2::zeros((20, 20));
    for i in 0..20 {
        data[[i, i]] = 1.0;
        if i > 0 {
            data[[i, i - 1]] = 1.0;
        }
    }

    let config = PlotConfig::with_title("Spy Plot Test");
    let matrix_plot = MatrixPlot::new(config);

    assert!(matrix_plot.spy(&data, &path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_eigenvalue_plot() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("eigenvalues.png");

    let eigenvalues = Array1::from_vec(vec![5.0, 3.0, 2.0, 1.0, 0.5]);

    let config = PlotConfig::with_title("Eigenvalue Plot Test");
    let matrix_plot = MatrixPlot::new(config);

    assert!(matrix_plot.eigenvalues(&eigenvalues, &path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_3d_scatter() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("scatter3d.png");

    let x = Array1::linspace(-5.0, 5.0, 20);
    let y = Array1::linspace(-5.0, 5.0, 20);
    let z = x.mapv(|v| v * v);

    let config = PlotConfig::with_title("3D Scatter Test");
    let plot3d = Plot3D::new(config);

    assert!(plot3d.scatter3d(&x, &y, &z, &path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_contour_plot() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("contour.png");

    let n = 20;
    let x = Array1::linspace(-2.0, 2.0, n);
    let y = Array1::linspace(-2.0, 2.0, n);

    let mut z = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            z[[i, j]] = x[j] * x[j] + y[i] * y[i];
        }
    }

    let config = PlotConfig::with_title("Contour Test");
    let plot3d = Plot3D::new(config);

    assert!(plot3d.contour(&x, &y, &z, &path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_speedup_curve() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("speedup.png");

    let scaling_data = vec![
        ScalingPoint {
            cores: 1,
            time: 10.0,
        },
        ScalingPoint {
            cores: 2,
            time: 5.5,
        },
        ScalingPoint {
            cores: 4,
            time: 3.0,
        },
    ];

    let config = PlotConfig::with_title("Speedup Test");
    let perf_plot = PerfPlot::new(config);

    assert!(perf_plot.speedup_curve(&scaling_data, 10.0, &path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_efficiency_curve() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("efficiency.png");

    let scaling_data = vec![
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

    let config = PlotConfig::with_title("Efficiency Test");
    let perf_plot = PerfPlot::new(config);

    assert!(perf_plot
        .efficiency_curve(&scaling_data, 10.0, &path)
        .is_ok());
    assert!(path.exists());
}

#[test]
fn test_benchmark_comparison() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("benchmarks.png");

    let benchmarks = vec![
        BenchmarkResult {
            name: "Test A".to_string(),
            time: 1.5,
            std_dev: Some(0.1),
        },
        BenchmarkResult {
            name: "Test B".to_string(),
            time: 2.3,
            std_dev: Some(0.15),
        },
    ];

    let config = PlotConfig::with_title("Benchmark Test");
    let perf_plot = PerfPlot::new(config);

    assert!(perf_plot.benchmark_comparison(&benchmarks, &path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_export_formats() {
    let temp_dir = setup_test_dir();

    assert_eq!(
        ExportFormat::from_extension(&PathBuf::from("test.png")).expect("PNG"),
        ExportFormat::Png
    );
    assert_eq!(
        ExportFormat::from_extension(&PathBuf::from("test.svg")).expect("SVG"),
        ExportFormat::Svg
    );
    assert_eq!(
        ExportFormat::from_extension(&PathBuf::from("test.html")).expect("HTML"),
        ExportFormat::Html
    );
    assert_eq!(
        ExportFormat::from_extension(&PathBuf::from("test.tex")).expect("TikZ"),
        ExportFormat::Tikz
    );

    assert!(ExportFormat::from_extension(&PathBuf::from("test.unknown")).is_err());
}

#[test]
fn test_tikz_export() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("tikz_test.tex");

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];

    assert!(Exporter::to_tikz(&x, &y, "Test", "X", "Y", &path).is_ok());
    assert!(path.exists());

    // Verify file contents
    let contents = std::fs::read_to_string(&path).expect("Failed to read file");
    assert!(contents.contains("\\begin{tikzpicture}"));
    assert!(contents.contains("\\begin{axis}"));
}

#[test]
fn test_color_from_hex() {
    let color = Color::from_hex("#FF0000").expect("Failed to parse hex");
    assert_eq!(color.r, 1.0);
    assert_eq!(color.g, 0.0);
    assert_eq!(color.b, 0.0);

    let color = Color::from_hex("#00FF00FF").expect("Failed to parse hex with alpha");
    assert_eq!(color.g, 1.0);
    assert_eq!(color.a, 1.0);

    assert!(Color::from_hex("#ZZZ").is_err());
}

#[test]
fn test_colormap() {
    let color = ColorMap::Viridis.get_color(0.5);
    assert!(color.r >= 0.0 && color.r <= 1.0);
    assert!(color.g >= 0.0 && color.g <= 1.0);
    assert!(color.b >= 0.0 && color.b <= 1.0);

    // Test all colormaps
    for &colormap in &[
        ColorMap::Viridis,
        ColorMap::Plasma,
        ColorMap::Inferno,
        ColorMap::Magma,
        ColorMap::Coolwarm,
        ColorMap::RdBu,
        ColorMap::Gray,
        ColorMap::Hot,
        ColorMap::Jet,
    ] {
        let c0 = colormap.get_color(0.0);
        let c1 = colormap.get_color(1.0);
        assert!(c0.r >= 0.0 && c0.r <= 1.0);
        assert!(c1.r >= 0.0 && c1.r <= 1.0);
    }
}

#[test]
fn test_plot_config() {
    let config = PlotConfig::with_title("Test")
        .with_x_label("X")
        .with_y_label("Y")
        .with_size(1024, 768)
        .expect("Failed to set size");

    assert_eq!(config.title, "Test");
    assert_eq!(config.x_axis.label, "X");
    assert_eq!(config.y_axis.label, "Y");
    assert_eq!(config.width, 1024);
    assert_eq!(config.height, 768);
}

#[test]
fn test_export_config() {
    let config = ExportConfig::new(ExportFormat::Png)
        .with_size(800, 600)
        .expect("Failed to set size")
        .with_dpi(150)
        .expect("Failed to set DPI");

    assert_eq!(config.format, ExportFormat::Png);
    assert_eq!(config.width, 800);
    assert_eq!(config.height, 600);
    assert_eq!(config.dpi, 150);
}

#[test]
fn test_svg_export() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("test.svg");

    let x = Array1::linspace(0.0, 10.0, 50);
    let y = x.mapv(|v: f64| v.sin());

    let config = PlotConfig::default();
    let mut plot = Plot2D::new(config);

    assert!(plot.line(&x, &y, "sin").is_ok());
    assert!(plot.save(&path).is_ok());
    assert!(path.exists());
}

#[test]
fn test_html_export() {
    let temp_dir = setup_test_dir();
    let path = temp_dir.path().join("test.html");

    let x = Array1::linspace(0.0, 10.0, 50);
    let y = x.mapv(|v: f64| v.sin());

    let config = PlotConfig::default();
    let mut plot = Plot2D::new(config);

    assert!(plot.line(&x, &y, "sin").is_ok());
    assert!(plot.save(&path).is_ok());
    assert!(path.exists());

    // Verify HTML structure
    let contents = std::fs::read_to_string(&path).expect("Failed to read HTML");
    assert!(contents.contains("<!DOCTYPE html>"));
    assert!(contents.contains("<svg"));
}
