# NumRS2 Visualization Guide

Advanced visualization tools for NumRS2 v0.2.0

## Overview

NumRS2's visualization module provides comprehensive plotting and data visualization capabilities for numerical computing, built entirely in pure Rust using the plotters library.

## Features

- **2D Plotting**: Line plots, scatter plots, bar charts, histograms, area plots
- **Statistical Visualization**: Histograms, box plots, violin plots, Q-Q plots, correlation heatmaps
- **Matrix Visualization**: Heatmaps, spy plots (sparse matrix patterns), eigenvalue plots
- **3D Visualization**: Surface plots, contour plots, 3D scatter plots
- **Performance Visualization**: Benchmark comparisons, speedup curves, efficiency plots
- **Multiple Export Formats**: PNG, SVG, HTML, LaTeX/TikZ

## Installation

Add the `visualization` feature to your Cargo.toml:

```toml
[dependencies]
numrs2 = { version = "0.2.0", features = ["visualization"] }
```

## Quick Start

```rust
use numrs2::viz::{Plot2D, PlotConfig};
use scirs2_core::ndarray::Array1;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create data
    let x = Array1::linspace(0.0, 2.0 * PI, 100);
    let y = x.mapv(|v| v.sin());

    // Create and configure plot
    let config = PlotConfig::with_title("Sine Wave")
        .with_x_label("x (radians)")
        .with_y_label("sin(x)");

    let mut plot = Plot2D::new(config);
    plot.line(&x, &y, "sin(x)")?;

    // Save to file
    plot.save("sine.png")?;

    Ok(())
}
```

## 2D Plotting

### Line Plots

```rust
use numrs2::viz::{Plot2D, PlotConfig, SeriesStyle, Color, LineStyle};

let config = PlotConfig::with_title("Multiple Series")
    .with_x_label("X")
    .with_y_label("Y");

let mut plot = Plot2D::new(config);

// Add simple line
plot.line(&x, &y1, "Series 1")?;

// Add styled line
let style = SeriesStyle {
    color: Color::RED,
    line_style: LineStyle::Dashed,
    ..Default::default()
};
plot.line_styled(&x, &y2, "Series 2", style)?;
```

### Scatter Plots

```rust
let mut plot = Plot2D::new(PlotConfig::default());
plot.scatter(&x, &y, "Data Points")?;
plot.save("scatter.png")?;
```

### Histograms

```rust
plot.histogram(&data, 30, "Distribution")?;
```

## Statistical Plots

### Histograms with Binning Strategies

```rust
use numrs2::viz::{StatPlot, BinStrategy};

let stat_plot = StatPlot::new(config);

// Auto binning (Sturges' formula)
stat_plot.histogram(&data, BinStrategy::Auto, &path)?;

// Scott's rule
stat_plot.histogram(&data, BinStrategy::Scott, &path)?;

// Freedman-Diaconis rule
stat_plot.histogram(&data, BinStrategy::FreedmanDiaconis, &path)?;

// Fixed number of bins
stat_plot.histogram(&data, BinStrategy::Fixed(50), &path)?;
```

### Box Plots

```rust
let stat_plot = StatPlot::new(config);
stat_plot.boxplot(&data, &path)?;
```

### Q-Q Plots

```rust
let stat_plot = StatPlot::new(config);
stat_plot.qqplot(&data, &path)?;
```

## Matrix Visualization

### Heatmaps

```rust
use numrs2::viz::{MatrixPlot, ColorMap};

let matrix_plot = MatrixPlot::new(config)
    .with_colormap(ColorMap::Viridis);

matrix_plot.heatmap(&matrix_data, &path)?;
```

### Spy Plots (Sparse Matrix Patterns)

```rust
let matrix_plot = MatrixPlot::new(config);
matrix_plot.spy(&sparse_matrix, &path)?;
```

### Eigenvalue Plots

```rust
let matrix_plot = MatrixPlot::new(config);
matrix_plot.eigenvalues(&eigenvalues, &path)?;
```

## 3D Visualization

### Surface Plots

```rust
use numrs2::viz::Plot3D;

let plot3d = Plot3D::new(config);
plot3d.surface(&x, &y, &z, &path)?;
```

### Contour Plots

```rust
let plot3d = Plot3D::new(config);
plot3d.contour(&x, &y, &z, &path)?;
```

### 3D Scatter Plots

```rust
let plot3d = Plot3D::new(config);
plot3d.scatter3d(&x, &y, &z, &path)?;
```

## Performance Visualization

### Speedup Curves

```rust
use numrs2::viz::{PerfPlot, ScalingPoint};

let scaling_data = vec![
    ScalingPoint { cores: 1, time: 10.0 },
    ScalingPoint { cores: 2, time: 5.5 },
    ScalingPoint { cores: 4, time: 3.0 },
    ScalingPoint { cores: 8, time: 1.8 },
];

let perf_plot = PerfPlot::new(config);
perf_plot.speedup_curve(&scaling_data, 10.0, &path)?;
```

### Efficiency Curves

```rust
perf_plot.efficiency_curve(&scaling_data, baseline_time, &path)?;
```

### Benchmark Comparisons

```rust
use numrs2::viz::BenchmarkResult;

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
];

perf_plot.benchmark_comparison(&benchmarks, &path)?;
```

## Color Maps

Available color maps for heatmaps and matrix visualizations:

- `Viridis` - Perceptually uniform (recommended)
- `Plasma` - Perceptually uniform
- `Inferno` - Perceptually uniform
- `Magma` - Perceptually uniform
- `Coolwarm` - Diverging color map
- `RdBu` - Red-blue diverging
- `Gray` - Grayscale
- `Hot` - Black-red-yellow-white
- `Jet` - Rainbow (use sparingly)

```rust
let matrix_plot = MatrixPlot::new(config)
    .with_colormap(ColorMap::Plasma);
```

## Export Formats

### PNG Export

```rust
plot.save("output.png")?;
```

### SVG Export

```rust
plot.save("output.svg")?;
```

### HTML Export

```rust
plot.save("output.html")?;
```

### LaTeX/TikZ Export

```rust
use numrs2::viz::Exporter;

Exporter::to_tikz_standalone(
    &x.to_vec(),
    &y.to_vec(),
    "Plot Title",
    "X Label",
    "Y Label",
    &path,
)?;
```

### Batch Export

```rust
use numrs2::viz::{Exporter, ExportFormat};

let formats = vec![
    ExportFormat::Png,
    ExportFormat::Svg,
    ExportFormat::Html,
];

Exporter::batch_export(&base_path, &formats, |path| {
    let mut plot = Plot2D::new(config.clone());
    plot.line(&x, &y, "data")?;
    plot.save(path)
})?;
```

## Configuration

### Plot Configuration

```rust
let config = PlotConfig::with_title("My Plot")
    .with_x_label("X Axis")
    .with_y_label("Y Axis")
    .with_size(1024, 768)?
    .with_backend(PlotBackend::Svg);
```

### Axis Configuration

```rust
let config = PlotConfig::default();
config.x_axis = AxisConfig::with_label("X")
    .with_range(0.0, 100.0)?
    .with_log_scale();
```

### Grid Configuration

```rust
config.grid.show_major = true;
config.grid.show_minor = true;
config.grid.major_color = Color::rgba(0.8, 0.8, 0.8, 0.5);
```

### Legend Configuration

```rust
config.legend.show = true;
config.legend.position = LegendPosition::TopRight;
config.legend.background = Color::rgba(1.0, 1.0, 1.0, 0.9);
```

## Examples

See `examples/visualization.rs` for comprehensive examples of all visualization types.

Run with:

```bash
cargo run --example visualization --features visualization
```

## Pure Rust Implementation

All visualization functionality is implemented in 100% pure Rust:

- **plotters**: Core plotting library
- **resvg**: SVG rendering
- **tiny-skia**: 2D graphics
- **No C/C++ dependencies**
- **Cross-platform compatibility**

## Performance Considerations

- **Memory**: Large plots may require significant memory
- **Resolution**: Higher DPI settings increase file size and rendering time
- **Format Choice**:
  - PNG: Fast, good for raster images
  - SVG: Scalable, good for publications
  - HTML: Interactive, good for exploration

## Best Practices

1. **Use appropriate plot types** for your data
2. **Choose perceptually uniform colormaps** (Viridis, Plasma) for scientific visualization
3. **Add clear labels** and titles to all plots
4. **Use temporary directories** (std::env::temp_dir()) for test outputs
5. **Handle errors properly** - all plotting functions return `VizResult`

## Integration with NumRS2

The visualization module integrates seamlessly with NumRS2's array operations:

```rust
use numrs2::prelude::*;
use numrs2::viz::{Plot2D, PlotConfig};

// NumRS2 array operations
let x = Array::linspace(0.0, 10.0, 100);
let y = x.mapv(|v| v.powi(2));

// Direct visualization
let mut plot = Plot2D::new(PlotConfig::default());
plot.line(&x, &y, "x²")?;
plot.save("parabola.png")?;
```

## Troubleshooting

### Common Issues

**Issue**: Plot file not created
- **Solution**: Check file path permissions and disk space

**Issue**: Empty or blank plot
- **Solution**: Verify data arrays are not empty and contain valid (finite) values

**Issue**: Compilation error with visualization
- **Solution**: Ensure `visualization` feature is enabled in Cargo.toml

## Contributing

Contributions to the visualization module are welcome! Please ensure:

- All code follows the no-unwrap policy
- Comprehensive tests are included
- Documentation includes examples
- Pure Rust implementation (no C/C++ dependencies)

## License

Apache-2.0

## References

- [plotters Documentation](https://docs.rs/plotters)
- [NumRS2 Main Documentation](https://docs.rs/numrs2)
- [SciRS2 Ecosystem](https://github.com/cool-japan/scirs2)

---

**NumRS2 v0.2.0** - Production-ready numerical computing for Rust
