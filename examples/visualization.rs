//! Comprehensive visualization examples for NumRS2
//!
//! This example demonstrates all the visualization capabilities of NumRS2.
//!
//! Run with: cargo run --example visualization --features visualization

use numrs2::viz::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::f64::consts::PI;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("NumRS2 Visualization Examples");
    println!("=============================\n");

    let output_dir = std::env::temp_dir().join("numrs2_viz_examples");
    std::fs::create_dir_all(&output_dir)?;
    println!("Output directory: {}\n", output_dir.display());

    // Example 1: Line Plot
    println!("1. Creating line plots...");
    line_plot_example(&output_dir)?;

    // Example 2: Scatter Plot
    println!("2. Creating scatter plots...");
    scatter_plot_example(&output_dir)?;

    // Example 3: Multiple Series
    println!("3. Creating multiple series plots...");
    multiple_series_example(&output_dir)?;

    // Example 4: Histogram
    println!("4. Creating histograms...");
    histogram_example(&output_dir)?;

    // Example 5: Box Plot
    println!("5. Creating box plots...");
    boxplot_example(&output_dir)?;

    // Example 6: Q-Q Plot
    println!("6. Creating Q-Q plots...");
    qqplot_example(&output_dir)?;

    // Example 7: Heatmap
    println!("7. Creating heatmaps...");
    heatmap_example(&output_dir)?;

    // Example 8: Matrix Spy Plot
    println!("8. Creating spy plots...");
    spy_plot_example(&output_dir)?;

    // Example 9: 3D Surface Plot
    println!("9. Creating 3D surface plots...");
    surface_plot_example(&output_dir)?;

    // Example 10: Contour Plot
    println!("10. Creating contour plots...");
    contour_plot_example(&output_dir)?;

    // Example 11: Performance Plots
    println!("11. Creating performance plots...");
    performance_plot_example(&output_dir)?;

    // Example 12: Export Formats
    println!("12. Demonstrating export formats...");
    export_formats_example(&output_dir)?;

    println!("\nAll examples completed successfully!");
    println!(
        "Check the output directory for generated plots: {}",
        output_dir.display()
    );

    Ok(())
}

/// Example 1: Line Plot
fn line_plot_example(output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let x = Array1::linspace(0.0, 2.0 * PI, 100);
    let y = x.mapv(|v| v.sin());

    let config = PlotConfig::with_title("Sine Wave")
        .with_x_label("x (radians)")
        .with_y_label("sin(x)");

    let mut plot = Plot2D::new(config);
    plot.line(&x, &y, "sin(x)")?;
    plot.save(output_dir.join("01_line_plot.png"))?;

    Ok(())
}

/// Example 2: Scatter Plot
fn scatter_plot_example(output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0)?;

    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| xi.powi(2) + rng.sample(normal))
        .collect();

    let x_arr = Array1::from_vec(x);
    let y_arr = Array1::from_vec(y);

    let config = PlotConfig::with_title("Scatter Plot with Noise")
        .with_x_label("x")
        .with_y_label("y");

    let mut plot = Plot2D::new(config);
    plot.scatter(&x_arr, &y_arr, "data")?;
    plot.save(output_dir.join("02_scatter_plot.png"))?;

    Ok(())
}

/// Example 3: Multiple Series
fn multiple_series_example(
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let x = Array1::linspace(0.0, 2.0 * PI, 100);
    let y1 = x.mapv(|v| v.sin());
    let y2 = x.mapv(|v| v.cos());
    let y3 = x.mapv(|v| (v * 2.0).sin());

    let config = PlotConfig::with_title("Multiple Trigonometric Functions")
        .with_x_label("x (radians)")
        .with_y_label("y");

    let mut plot = Plot2D::new(config);
    plot.line(&x, &y1, "sin(x)")?;

    let style2 = SeriesStyle {
        color: Color::RED,
        ..Default::default()
    };
    plot.line_styled(&x, &y2, "cos(x)", style2)?;

    let style3 = SeriesStyle {
        color: Color::GREEN,
        line_style: LineStyle::Dashed,
        ..Default::default()
    };
    plot.line_styled(&x, &y3, "sin(2x)", style3)?;

    plot.save(output_dir.join("03_multiple_series.png"))?;

    Ok(())
}

/// Example 4: Histogram
fn histogram_example(output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0)?;

    let data: Vec<f64> = (0..1000).map(|_| rng.sample(normal)).collect();
    let data_arr = Array1::from_vec(data);

    let config = PlotConfig::with_title("Histogram of Normal Distribution")
        .with_x_label("Value")
        .with_y_label("Frequency");

    let stat_plot = StatPlot::new(config);
    stat_plot.histogram(
        &data_arr,
        BinStrategy::Sturges,
        &output_dir.join("04_histogram.png"),
    )?;

    Ok(())
}

/// Example 5: Box Plot
fn boxplot_example(output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(50.0, 10.0)?;

    let data: Vec<f64> = (0..100).map(|_| rng.sample(normal)).collect();
    let data_arr = Array1::from_vec(data);

    let config = PlotConfig::with_title("Box Plot Example").with_y_label("Value");

    let stat_plot = StatPlot::new(config);
    stat_plot.boxplot(&data_arr, &output_dir.join("05_boxplot.png"))?;

    Ok(())
}

/// Example 6: Q-Q Plot
fn qqplot_example(output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0)?;

    let data: Vec<f64> = (0..200).map(|_| rng.sample(normal)).collect();
    let data_arr = Array1::from_vec(data);

    let config = PlotConfig::with_title("Q-Q Plot: Normal Distribution");

    let stat_plot = StatPlot::new(config);
    stat_plot.qqplot(&data_arr, &output_dir.join("06_qqplot.png"))?;

    Ok(())
}

/// Example 7: Heatmap
fn heatmap_example(output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let rows = 20;
    let cols = 20;
    let mut data = Array2::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            let x = i as f64 / rows as f64 * 2.0 * PI;
            let y = j as f64 / cols as f64 * 2.0 * PI;
            data[[i, j]] = (x.sin() * y.cos()).abs();
        }
    }

    let config = PlotConfig::with_title("Heatmap Example");

    let matrix_plot = MatrixPlot::new(config).with_colormap(ColorMap::Viridis);
    matrix_plot.heatmap(&data, &output_dir.join("07_heatmap.png"))?;

    Ok(())
}

/// Example 8: Spy Plot
fn spy_plot_example(output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let size = 30;
    let mut data = Array2::zeros((size, size));

    // Create a tridiagonal matrix pattern
    for i in 0..size {
        data[[i, i]] = 1.0;
        if i > 0 {
            data[[i, i - 1]] = 1.0;
        }
        if i < size - 1 {
            data[[i, i + 1]] = 1.0;
        }
    }

    // Add some random non-zeros
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..20 {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        data[[i, j]] = 1.0;
    }

    let config = PlotConfig::with_title("Spy Plot: Sparse Matrix Pattern");

    let matrix_plot = MatrixPlot::new(config);
    matrix_plot.spy(&data, &output_dir.join("08_spy_plot.png"))?;

    Ok(())
}

/// Example 9: Surface Plot
fn surface_plot_example(output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let n = 30;
    let x = Array1::linspace(-3.0, 3.0, n);
    let y = Array1::linspace(-3.0, 3.0, n);

    let mut z = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let xi = x[j];
            let yi = y[i];
            z[[i, j]] = (-(xi * xi + yi * yi) / 2.0_f64).exp();
        }
    }

    let config = PlotConfig::with_title("3D Surface: Gaussian");

    let plot3d = Plot3D::new(config);
    plot3d.surface(&x, &y, &z, &output_dir.join("09_surface.png"))?;

    Ok(())
}

/// Example 10: Contour Plot
fn contour_plot_example(output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let n = 50;
    let x = Array1::linspace(-2.0, 2.0, n);
    let y = Array1::linspace(-2.0, 2.0, n);

    let mut z = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let xi = x[j];
            let yi = y[i];
            z[[i, j]] = xi * xi - yi * yi;
        }
    }

    let config = PlotConfig::with_title("Contour Plot: f(x,y) = x² - y²")
        .with_x_label("x")
        .with_y_label("y");

    let plot3d = Plot3D::new(config);
    plot3d.contour(&x, &y, &z, &output_dir.join("10_contour.png"))?;

    Ok(())
}

/// Example 11: Performance Plots
fn performance_plot_example(
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Speedup curve
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
        ScalingPoint {
            cores: 8,
            time: 1.8,
        },
        ScalingPoint {
            cores: 16,
            time: 1.2,
        },
    ];

    let config = PlotConfig::with_title("Speedup Curve");
    let perf_plot = PerfPlot::new(config);
    perf_plot.speedup_curve(&scaling_data, 10.0, &output_dir.join("11_speedup.png"))?;

    // Efficiency curve
    let config = PlotConfig::with_title("Parallel Efficiency");
    let perf_plot = PerfPlot::new(config);
    perf_plot.efficiency_curve(&scaling_data, 10.0, &output_dir.join("11_efficiency.png"))?;

    // Benchmark comparison
    let benchmarks = vec![
        BenchmarkResult {
            name: "Algorithm A".to_string(),
            time: 1.5,
            std_dev: Some(0.1),
        },
        BenchmarkResult {
            name: "Algorithm B".to_string(),
            time: 2.3,
            std_dev: Some(0.15),
        },
        BenchmarkResult {
            name: "Algorithm C".to_string(),
            time: 0.8,
            std_dev: Some(0.05),
        },
    ];

    let config = PlotConfig::with_title("Benchmark Comparison");
    let perf_plot = PerfPlot::new(config);
    perf_plot.benchmark_comparison(&benchmarks, &output_dir.join("11_benchmarks.png"))?;

    Ok(())
}

/// Example 12: Export Formats
fn export_formats_example(output_dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let x = Array1::linspace(0.0, 2.0 * PI, 50);
    let y = x.mapv(|v| v.sin());

    // PNG export
    let config = PlotConfig::with_title("Export Demo - PNG")
        .with_x_label("x")
        .with_y_label("sin(x)");
    let mut plot = Plot2D::new(config);
    plot.line(&x, &y, "sin(x)")?;
    plot.save(output_dir.join("12_export.png"))?;

    // SVG export
    let config = PlotConfig::with_title("Export Demo - SVG")
        .with_x_label("x")
        .with_y_label("sin(x)");
    let mut plot = Plot2D::new(config);
    plot.line(&x, &y, "sin(x)")?;
    plot.save(output_dir.join("12_export.svg"))?;

    // HTML export
    let config = PlotConfig::with_title("Export Demo - HTML")
        .with_x_label("x")
        .with_y_label("sin(x)");
    let mut plot = Plot2D::new(config);
    plot.line(&x, &y, "sin(x)")?;
    plot.save(output_dir.join("12_export.html"))?;

    // TikZ export
    Exporter::to_tikz_standalone(
        &x.to_vec(),
        &y.to_vec(),
        "Export Demo - TikZ",
        "x",
        "sin(x)",
        &output_dir.join("12_export.tex"),
    )?;

    Ok(())
}
