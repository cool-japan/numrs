# NumRS Statistical Functions

The NumRS statistics module provides a rich set of statistical functions for data analysis. These functions include descriptive statistics, correlation analysis, histograms, and various methods for analyzing distributions.

## Key Features

- Descriptive statistics (mean, variance, std, min, max)
- Correlation and covariance calculations
- Histogram generation (1D and 2D)
- Percentile and quantile calculations
- Binning and counting operations
- Weighted statistics
- Array-based API similar to NumPy's statistics functions

## Basic Usage

### Descriptive Statistics

```rust
use numrs2::prelude::*;

// Create a sample array
let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

// Basic statistics using the Statistics trait
let mean = data.mean();          // 3.0
let variance = data.var();       // 2.0
let std_dev = data.std();        // 1.414...
let minimum = data.min();        // 1.0
let maximum = data.max();        // 5.0

// Peak-to-peak range
let range = ptp(&data, None)?;   // 4.0
```

### Percentiles and Quantiles

```rust
use numrs2::stats::{percentile, quantile};

// Create data array
let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

// Calculate percentiles (0-100 scale)
let p_points = Array::from_vec(vec![0.0, 25.0, 50.0, 75.0, 100.0]);
let p_values = percentile(&data, &p_points, Some("linear"))?;
// -> [1.0, 2.0, 3.0, 4.0, 5.0]

// Calculate quantiles (0-1 scale)
let q_points = Array::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
let q_values = quantile(&data, &q_points, Some("linear"))?;
// -> [1.0, 2.0, 3.0, 4.0, 5.0]

// Different interpolation methods are available:
// - "linear": Linear interpolation between points
// - "lower": Use the lower data point
// - "higher": Use the higher data point
// - "nearest": Use the nearest data point
// - "midpoint": Use the midpoint between adjacent data points
```

### Correlation and Covariance

```rust
use numrs2::stats::{cov, corrcoef};

// Create two data arrays
let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
let y = Array::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);

// Calculate covariance
let covariance = cov(&x, &y)?;  // -2.5

// Calculate correlation coefficient
let correlation = corrcoef(&x, &y)?;  // -1.0 (perfect negative correlation)
```

### Histograms

```rust
use numrs2::stats::{histogram, histogram2d};

// Create sample data
let data = Array::from_vec(vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);

// Create a histogram with 5 bins
let (hist, bin_edges) = histogram(&data, 5, None, None)?;

// Create a 2D histogram
let x = Array::from_vec(vec![1.0, 2.0, 2.5, 3.0, 4.0, 5.0]);
let y = Array::from_vec(vec![2.0, 3.0, 3.5, 4.0, 5.0, 6.0]);

// Create a 2D histogram with 3x3 bins
let (hist2d, x_edges, y_edges) = histogram2d(&x, &y, 3, None, None)?;
```

### Binning and Counting

```rust
use numrs2::stats::{bincount, digitize};

// Count occurrences of integers
let data = Array::from_vec(vec![0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]);
let counts = bincount(&data, None, Some(5))?;
// -> [1, 2, 3, 4, 0]

// Assign values to bins
let values = Array::from_vec(vec![0.2, 1.4, 2.5, 3.7, 4.8]);
let bins = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
let indices = digitize(&values, &bins, Some(false))?;
// Assigns each value to a bin
```

### Weighted Statistics

```rust
use numrs2::stats::average;

// Create data and weights
let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
let weights = Array::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);

// Calculate weighted average
let weighted_avg = average(&data, Some(&weights), None, None)?;
// Regular average for comparison
let avg = average(&data, None, None, None)?;

// Create weighted histogram
let (hist, bin_edges) = histogram(&data, 5, None, Some(&weights))?;
```

## Examples

For detailed examples, see:
- `statistics_example.rs`: Comprehensive demonstration of statistical functions

## Implementation Details

- The statistical functions in NumRS are designed to be numerically stable
- Functions support various data types through generic implementations
- Most functions include optional parameters for customization
- Where applicable, functions can operate along specific array axes
- The implementation aims to match NumPy's behavior for compatibility

## Performance Notes

- Functions are optimized for moderate-sized datasets
- For very large datasets, consider using parallel processing with the `parallel_optimize` module
- Some statistical operations require sorting, which can affect performance for large arrays
- Weighted statistics may have additional overhead compared to unweighted operations