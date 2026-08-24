//! Histogram functions
//!
//! This module provides histogram calculation and related utilities:
//! - histogram: Calculate a 1D histogram
//! - histogram_bin_edges: Compute bin edges without computing the histogram
//! - histogram2d: Calculate a 2D histogram
//! - histogram_dd: Calculate a multi-dimensional histogram
//! - bincount: Count occurrences of each value
//! - digitize: Return the indices of the bins for each value
//! - BinSpec: Enum for specifying bin strategies
//! - HistBins: Helper enum for histogram2d bins

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};
use scirs2_core::parallel_ops::*;

use super::basic::PARALLEL_THRESHOLD;

/// Calculate a histogram of a dataset with parallel processing for large arrays
///
/// # Parameters
///
/// * `a` - Input array
/// * `bins` - Number of bins
/// * `range` - Optional tuple of (min, max) to use for bin edges
/// * `weights` - Optional array of weights for each value
///
/// # Returns
///
/// A tuple of (histogram counts, bin edges)
pub fn histogram<T: Float + Clone + NumCast + std::fmt::Display + Send + Sync + 'static>(
    a: &Array<T>,
    bins: usize,
    range: Option<(T, T)>,
    weights: Option<&Array<T>>,
) -> Result<(Array<T>, Array<T>)> {
    use super::basic::Statistics;

    let data = a.to_vec();
    if data.is_empty() || bins == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute histogram of an empty array or with zero bins".to_string(),
        ));
    }

    // Get min and max values - either from range parameter or from data
    let (min_val, max_val) = match range {
        Some((min, max)) => {
            if min >= max {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Range ({}, {}) is invalid: min must be less than max",
                    min, max
                )));
            }
            (min, max)
        }
        None => (a.min(), a.max()),
    };

    // Create bin edges
    let step = (max_val - min_val) / T::from(bins).expect("bins should be representable");
    let mut bin_edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        bin_edges.push(min_val + step * T::from(i).expect("index should be representable"));
    }

    // Count values in each bin with optional weights using parallel processing for large datasets
    let mut counts = vec![T::zero(); bins];

    // Precompute inverse step for faster bin computation (multiply is faster than divide)
    let inv_step = T::one() / step;

    if data.len() >= PARALLEL_THRESHOLD {
        // Use chunked parallel processing for large datasets
        // This is much more efficient than per-element parallelism because:
        // 1. Memory: O(num_chunks * bins) instead of O(n * bins)
        // 2. Cache: Better locality with chunk-based processing
        // 3. Reduction: Fewer partial results to merge

        // Choose chunk size for good cache utilization (aim for ~64KB per chunk)
        let chunk_size = (64 * 1024 / std::mem::size_of::<T>())
            .max(1024)
            .min(data.len());

        if let Some(w) = weights {
            let weights_data = w.to_vec();

            if weights_data.len() != data.len() {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![data.len()],
                    actual: vec![weights_data.len()],
                });
            }

            // Parallel chunked reduction with weights
            let partial_histograms: Vec<Vec<T>> = data
                .par_chunks(chunk_size)
                .zip(weights_data.par_chunks(chunk_size))
                .map(|(chunk, weight_chunk)| {
                    let mut local_counts = vec![T::zero(); bins];
                    for (&val, &weight) in chunk.iter().zip(weight_chunk.iter()) {
                        if val >= min_val && val <= max_val {
                            let bin_idx = if val == max_val {
                                bins - 1
                            } else {
                                ((val - min_val) * inv_step)
                                    .to_usize()
                                    .expect("bin index should be convertible")
                                    .min(bins - 1)
                            };
                            local_counts[bin_idx] = local_counts[bin_idx] + weight;
                        }
                    }
                    local_counts
                })
                .collect();

            // Reduce partial histograms - this is O(num_chunks * bins) which is small
            for partial in partial_histograms {
                for (i, &count) in partial.iter().enumerate() {
                    counts[i] = counts[i] + count;
                }
            }
        } else {
            // No weights - parallel chunked counting
            let partial_histograms: Vec<Vec<T>> = data
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_counts = vec![T::zero(); bins];
                    for &val in chunk {
                        if val >= min_val && val <= max_val {
                            let bin_idx = if val == max_val {
                                bins - 1
                            } else {
                                ((val - min_val) * inv_step)
                                    .to_usize()
                                    .expect("bin index should be convertible")
                                    .min(bins - 1)
                            };
                            local_counts[bin_idx] = local_counts[bin_idx] + T::one();
                        }
                    }
                    local_counts
                })
                .collect();

            // Reduce partial histograms
            for partial in partial_histograms {
                for (i, &count) in partial.iter().enumerate() {
                    counts[i] = counts[i] + count;
                }
            }
        }
    } else {
        // Use sequential processing for small datasets
        if let Some(w) = weights {
            let weights_data = w.to_vec();

            if weights_data.len() != data.len() {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![data.len()],
                    actual: vec![weights_data.len()],
                });
            }

            for (i, &val) in data.iter().enumerate() {
                if val >= min_val && val <= max_val {
                    let bin_idx = if val == max_val {
                        bins - 1
                    } else {
                        ((val - min_val) * inv_step)
                            .to_usize()
                            .expect("bin index should be convertible")
                            .min(bins - 1)
                    };
                    counts[bin_idx] = counts[bin_idx] + weights_data[i];
                }
            }
        } else {
            // No weights - just count occurrences
            for &val in &data {
                if val >= min_val && val <= max_val {
                    let bin_idx = if val == max_val {
                        bins - 1
                    } else {
                        ((val - min_val) * inv_step)
                            .to_usize()
                            .expect("bin index should be convertible")
                            .min(bins - 1)
                    };
                    counts[bin_idx] = counts[bin_idx] + T::one();
                }
            }
        }
    }

    Ok((Array::from_vec(counts), Array::from_vec(bin_edges)))
}

/// Compute the bin edges for a histogram without computing the histogram itself
///
/// This function computes the bin edges that would be used by `histogram()`,
/// which is useful when you need the edges for multiple histograms with the
/// same binning scheme.
///
/// # Parameters
///
/// * `a` - Input data array
/// * `bins` - Either a number of equal-width bins or a string strategy:
///   - "auto": Use the maximum of 'sturges' and 'fd' estimators
///   - "sqrt": Square root of the number of elements
///   - "sturges": Sturges formula
///   - "fd": Freedman-Diaconis rule
///   - "rice": Rice rule
///   - "scott": Scott's normal reference rule
///   - "doane": Doane's formula
///   - or an integer number of bins
/// * `range` - Optional (min, max) range for the edges
///
/// # Returns
///
/// An array of bin edges with length `bins + 1`
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
/// use numrs2::stats::{histogram_bin_edges, BinSpec};
///
/// let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let edges = histogram_bin_edges(&data, BinSpec::Count(5), None).expect("bin_edges should succeed");
/// assert_eq!(edges.size(), 6); // 5 bins = 6 edges
/// ```
#[derive(Clone, Debug)]
pub enum BinSpec {
    /// Fixed number of bins
    Count(usize),
    /// Automatic bin selection strategy
    Auto,
    /// Square root rule
    Sqrt,
    /// Sturges formula
    Sturges,
    /// Freedman-Diaconis rule
    Fd,
    /// Rice rule
    Rice,
    /// Scott's normal reference rule
    Scott,
    /// Doane's formula
    Doane,
}

impl From<usize> for BinSpec {
    fn from(n: usize) -> Self {
        BinSpec::Count(n)
    }
}

impl From<&str> for BinSpec {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "auto" => BinSpec::Auto,
            "sqrt" => BinSpec::Sqrt,
            "sturges" => BinSpec::Sturges,
            "fd" | "freedman-diaconis" => BinSpec::Fd,
            "rice" => BinSpec::Rice,
            "scott" => BinSpec::Scott,
            "doane" => BinSpec::Doane,
            _ => BinSpec::Auto,
        }
    }
}

pub fn histogram_bin_edges<T: Float + Clone + NumCast + std::fmt::Display + Send + Sync>(
    a: &Array<T>,
    bins: impl Into<BinSpec>,
    range: Option<(T, T)>,
) -> Result<Array<T>> {
    let data = a.to_vec();
    if data.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute bin edges for an empty array".to_string(),
        ));
    }

    let bins_spec = bins.into();

    // Get min and max values - either from range parameter or from data
    let (min_val, max_val) = match range {
        Some((min, max)) => {
            if min >= max {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Range ({}, {}) is invalid: min must be less than max",
                    min, max
                )));
            }
            (min, max)
        }
        None => {
            let min = data.iter().cloned().fold(T::infinity(), |a, b| a.min(b));
            let max = data
                .iter()
                .cloned()
                .fold(T::neg_infinity(), |a, b| a.max(b));
            (min, max)
        }
    };

    // Calculate number of bins based on the specification
    let n_bins = match bins_spec {
        BinSpec::Count(n) => {
            if n == 0 {
                return Err(NumRs2Error::InvalidOperation(
                    "Number of bins must be positive".to_string(),
                ));
            }
            n
        }
        BinSpec::Auto => {
            // Use maximum of sturges and fd
            let sturges = compute_sturges_bins(data.len());
            let fd = compute_fd_bins(&data, min_val, max_val);
            sturges.max(fd)
        }
        BinSpec::Sqrt => {
            // Square root rule: ceil(sqrt(n))
            let n = data.len() as f64;
            n.sqrt().ceil() as usize
        }
        BinSpec::Sturges => compute_sturges_bins(data.len()),
        BinSpec::Fd => compute_fd_bins(&data, min_val, max_val),
        BinSpec::Rice => {
            // Rice rule: ceil(2 * n^(1/3))
            let n = data.len() as f64;
            (2.0 * n.powf(1.0 / 3.0)).ceil() as usize
        }
        BinSpec::Scott => compute_scott_bins(&data, min_val, max_val),
        BinSpec::Doane => compute_doane_bins(&data),
    };

    // Ensure at least 1 bin
    let n_bins = n_bins.max(1);

    // Create bin edges
    let step = (max_val - min_val) / T::from(n_bins).expect("n_bins should be representable");
    let mut bin_edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        bin_edges.push(min_val + step * T::from(i).expect("index should be representable"));
    }

    Ok(Array::from_vec(bin_edges))
}

/// Compute number of bins using Sturges' formula
fn compute_sturges_bins(n: usize) -> usize {
    // k = ceil(log2(n) + 1)
    let n_f = n as f64;
    (n_f.log2() + 1.0).ceil() as usize
}

/// Compute number of bins using Freedman-Diaconis rule
fn compute_fd_bins<T: Float + Clone + NumCast>(data: &[T], min_val: T, max_val: T) -> usize {
    if data.len() < 4 {
        return 1;
    }

    // Sort data for IQR calculation
    let mut sorted: Vec<T> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate IQR
    let n = sorted.len();
    let q1_idx = n / 4;
    let q3_idx = (3 * n) / 4;
    let iqr = sorted[q3_idx] - sorted[q1_idx];

    if iqr == T::zero() {
        return 1;
    }

    // bin width h = 2 * IQR / n^(1/3)
    let n_f = T::from(n).expect("n should be representable");
    let h = T::from(2.0).expect("2.0 should be representable") * iqr
        / n_f.powf(T::from(1.0 / 3.0).expect("1/3 should be representable"));

    if h == T::zero() {
        return 1;
    }

    // number of bins = ceil((max - min) / h)
    let range = max_val - min_val;
    ((range / h)
        .to_f64()
        .expect("range/h should be convertible")
        .ceil() as usize)
        .max(1)
}

/// Compute number of bins using Scott's normal reference rule
fn compute_scott_bins<T: Float + Clone + NumCast>(data: &[T], min_val: T, max_val: T) -> usize {
    if data.len() < 2 {
        return 1;
    }

    // Calculate standard deviation
    let n = data.len();
    let n_f = T::from(n).expect("n should be representable");
    let mean: T = data.iter().cloned().fold(T::zero(), |a, b| a + b) / n_f;
    let variance: T = data
        .iter()
        .cloned()
        .map(|x| (x - mean) * (x - mean))
        .fold(T::zero(), |a, b| a + b)
        / n_f;
    let std_dev = variance.sqrt();

    if std_dev == T::zero() {
        return 1;
    }

    // bin width h = 3.49 * std / n^(1/3)
    let h = T::from(3.49).expect("3.49 should be representable") * std_dev
        / n_f.powf(T::from(1.0 / 3.0).expect("1/3 should be representable"));

    if h == T::zero() {
        return 1;
    }

    // number of bins = ceil((max - min) / h)
    let range = max_val - min_val;
    ((range / h)
        .to_f64()
        .expect("range/h should be convertible")
        .ceil() as usize)
        .max(1)
}

/// Compute number of bins using Doane's formula
fn compute_doane_bins<T: Float + Clone + NumCast>(data: &[T]) -> usize {
    if data.len() < 3 {
        return 1;
    }

    let n = data.len();
    let n_f = n as f64;

    // Calculate skewness
    let mean: f64 = data
        .iter()
        .map(|x| x.to_f64().expect("value should be convertible to f64"))
        .sum::<f64>()
        / n_f;

    let variance: f64 = data
        .iter()
        .map(|x| {
            let d = x.to_f64().expect("value should be convertible to f64") - mean;
            d * d
        })
        .sum::<f64>()
        / n_f;

    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return 1;
    }

    let skewness: f64 = data
        .iter()
        .map(|x| {
            let d = (x.to_f64().expect("value should be convertible to f64") - mean) / std_dev;
            d * d * d
        })
        .sum::<f64>()
        / n_f;

    // Doane's formula: k = 1 + log2(n) + log2(1 + |g1| / sigma_g1)
    // where sigma_g1 = sqrt(6(n-2) / ((n+1)(n+3)))
    let sigma_g1 = (6.0 * (n_f - 2.0) / ((n_f + 1.0) * (n_f + 3.0))).sqrt();

    let k = 1.0 + n_f.log2() + (1.0 + skewness.abs() / sigma_g1).log2();

    k.ceil() as usize
}

/// Helper enum to specify bins for histogram2d
pub enum HistBins {
    Single(usize),
    Tuple(usize, usize),
}

impl From<usize> for HistBins {
    fn from(val: usize) -> Self {
        HistBins::Single(val)
    }
}

impl From<(usize, usize)> for HistBins {
    fn from(val: (usize, usize)) -> Self {
        HistBins::Tuple(val.0, val.1)
    }
}

/// Calculate a 2D histogram of a dataset
///
/// # Parameters
///
/// * `x` - Input array for x coordinates
/// * `y` - Input array for y coordinates
/// * `bins` - Either a tuple (nx, ny) to specify bins in each dimension,
///   or a single value to use the same number of bins in both dimensions
/// * `range` - Optional tuple ((xmin, xmax), (ymin, ymax)) to use for bin edges
/// * `weights` - Optional array of weights for each value
///
/// # Returns
///
/// A tuple of (histogram counts, x_edges, y_edges)
pub fn histogram2d<T: Float + Clone + NumCast + std::fmt::Display + Send + Sync>(
    x: &Array<T>,
    y: &Array<T>,
    bins: impl Into<HistBins>,
    range: Option<((T, T), (T, T))>,
    weights: Option<&Array<T>>,
) -> Result<(Array<T>, Array<T>, Array<T>)> {
    let bins_val = bins.into();
    let (x_bins, y_bins) = match bins_val {
        HistBins::Single(n) => (n, n),
        HistBins::Tuple(nx, ny) => (nx, ny),
    };

    // Check inputs
    let x_data = x.to_vec();
    let y_data = y.to_vec();

    if x_data.len() != y_data.len() {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![x_data.len()],
            actual: vec![y_data.len()],
        });
    }

    if x_data.is_empty() || x_bins == 0 || y_bins == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute histogram2d with empty arrays or zero bins".to_string(),
        ));
    }

    // Get min and max values - either from range parameter or from data
    let (x_min, x_max) = match range {
        Some(((min, max), _)) => {
            if min >= max {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "X range ({}, {}) is invalid: min must be less than max",
                    min, max
                )));
            }
            (min, max)
        }
        None => {
            let x_data = x.to_vec();
            let x_min = x_data
                .iter()
                .fold(x_data[0], |acc, &val| if val < acc { val } else { acc });
            let x_max = x_data
                .iter()
                .fold(x_data[0], |acc, &val| if val > acc { val } else { acc });
            (x_min, x_max)
        }
    };

    let (y_min, y_max) = match range {
        Some((_, (min, max))) => {
            if min >= max {
                return Err(NumRs2Error::InvalidOperation(format!(
                    "Y range ({}, {}) is invalid: min must be less than max",
                    min, max
                )));
            }
            (min, max)
        }
        None => {
            let y_data = y.to_vec();
            let y_min = y_data
                .iter()
                .fold(y_data[0], |acc, &val| if val < acc { val } else { acc });
            let y_max = y_data
                .iter()
                .fold(y_data[0], |acc, &val| if val > acc { val } else { acc });
            (y_min, y_max)
        }
    };

    // Create bin edges
    let x_step = (x_max - x_min) / T::from(x_bins).expect("x_bins should be representable");
    let mut x_edges = Vec::with_capacity(x_bins + 1);
    for i in 0..=x_bins {
        x_edges.push(x_min + x_step * T::from(i).expect("index should be representable"));
    }

    let y_step = (y_max - y_min) / T::from(y_bins).expect("y_bins should be representable");
    let mut y_edges = Vec::with_capacity(y_bins + 1);
    for i in 0..=y_bins {
        y_edges.push(y_min + y_step * T::from(i).expect("index should be representable"));
    }

    // Precompute inverse steps for faster bin computation (multiply is faster than divide)
    let x_inv_step = T::one() / x_step;
    let y_inv_step = T::one() / y_step;

    // Total bin count for flat histogram
    let total_bins = x_bins * y_bins;

    // Initialize flat histogram (row-major order)
    let flat_hist = if x_data.len() >= PARALLEL_THRESHOLD {
        // Use chunked parallel processing for large datasets
        let chunk_size = (64 * 1024 / (2 * std::mem::size_of::<T>()))
            .max(1024)
            .min(x_data.len());

        if let Some(w) = weights {
            let weights_data = w.to_vec();

            if weights_data.len() != x_data.len() {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![x_data.len()],
                    actual: vec![weights_data.len()],
                });
            }

            // Create index array for chunking
            let indices: Vec<usize> = (0..x_data.len()).collect();

            // Parallel chunked reduction with weights
            let partial_histograms: Vec<Vec<T>> = indices
                .par_chunks(chunk_size)
                .map(|idx_chunk| {
                    let mut local_hist = vec![T::zero(); total_bins];
                    for &i in idx_chunk {
                        let x_val = x_data[i];
                        let y_val = y_data[i];
                        let weight = weights_data[i];

                        if x_val >= x_min && x_val <= x_max && y_val >= y_min && y_val <= y_max {
                            let x_idx = if x_val == x_max {
                                x_bins - 1
                            } else {
                                ((x_val - x_min) * x_inv_step)
                                    .to_usize()
                                    .expect("x bin index should be convertible")
                                    .min(x_bins - 1)
                            };
                            let y_idx = if y_val == y_max {
                                y_bins - 1
                            } else {
                                ((y_val - y_min) * y_inv_step)
                                    .to_usize()
                                    .expect("y bin index should be convertible")
                                    .min(y_bins - 1)
                            };
                            local_hist[x_idx * y_bins + y_idx] =
                                local_hist[x_idx * y_bins + y_idx] + weight;
                        }
                    }
                    local_hist
                })
                .collect();

            // Reduce partial histograms
            let mut result = vec![T::zero(); total_bins];
            for partial in partial_histograms {
                for (i, &count) in partial.iter().enumerate() {
                    result[i] = result[i] + count;
                }
            }
            result
        } else {
            // No weights - parallel chunked counting
            let indices: Vec<usize> = (0..x_data.len()).collect();

            let partial_histograms: Vec<Vec<T>> = indices
                .par_chunks(chunk_size)
                .map(|idx_chunk| {
                    let mut local_hist = vec![T::zero(); total_bins];
                    for &i in idx_chunk {
                        let x_val = x_data[i];
                        let y_val = y_data[i];

                        if x_val >= x_min && x_val <= x_max && y_val >= y_min && y_val <= y_max {
                            let x_idx = if x_val == x_max {
                                x_bins - 1
                            } else {
                                ((x_val - x_min) * x_inv_step)
                                    .to_usize()
                                    .expect("x bin index should be convertible")
                                    .min(x_bins - 1)
                            };
                            let y_idx = if y_val == y_max {
                                y_bins - 1
                            } else {
                                ((y_val - y_min) * y_inv_step)
                                    .to_usize()
                                    .expect("y bin index should be convertible")
                                    .min(y_bins - 1)
                            };
                            local_hist[x_idx * y_bins + y_idx] =
                                local_hist[x_idx * y_bins + y_idx] + T::one();
                        }
                    }
                    local_hist
                })
                .collect();

            // Reduce partial histograms
            let mut result = vec![T::zero(); total_bins];
            for partial in partial_histograms {
                for (i, &count) in partial.iter().enumerate() {
                    result[i] = result[i] + count;
                }
            }
            result
        }
    } else {
        // Sequential processing for small datasets
        let mut hist = vec![T::zero(); total_bins];

        if let Some(w) = weights {
            let weights_data = w.to_vec();

            if weights_data.len() != x_data.len() {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![x_data.len()],
                    actual: vec![weights_data.len()],
                });
            }

            for i in 0..x_data.len() {
                let x_val = x_data[i];
                let y_val = y_data[i];
                let weight = weights_data[i];

                if x_val >= x_min && x_val <= x_max && y_val >= y_min && y_val <= y_max {
                    let x_idx = if x_val == x_max {
                        x_bins - 1
                    } else {
                        ((x_val - x_min) * x_inv_step)
                            .to_usize()
                            .expect("x bin index should be convertible")
                            .min(x_bins - 1)
                    };
                    let y_idx = if y_val == y_max {
                        y_bins - 1
                    } else {
                        ((y_val - y_min) * y_inv_step)
                            .to_usize()
                            .expect("y bin index should be convertible")
                            .min(y_bins - 1)
                    };
                    hist[x_idx * y_bins + y_idx] = hist[x_idx * y_bins + y_idx] + weight;
                }
            }
        } else {
            for i in 0..x_data.len() {
                let x_val = x_data[i];
                let y_val = y_data[i];

                if x_val >= x_min && x_val <= x_max && y_val >= y_min && y_val <= y_max {
                    let x_idx = if x_val == x_max {
                        x_bins - 1
                    } else {
                        ((x_val - x_min) * x_inv_step)
                            .to_usize()
                            .expect("x bin index should be convertible")
                            .min(x_bins - 1)
                    };
                    let y_idx = if y_val == y_max {
                        y_bins - 1
                    } else {
                        ((y_val - y_min) * y_inv_step)
                            .to_usize()
                            .expect("y bin index should be convertible")
                            .min(y_bins - 1)
                    };
                    hist[x_idx * y_bins + y_idx] = hist[x_idx * y_bins + y_idx] + T::one();
                }
            }
        }
        hist
    };

    Ok((
        Array::from_vec_shape(flat_hist, &[x_bins, y_bins])?,
        Array::from_vec(x_edges),
        Array::from_vec(y_edges),
    ))
}

/// Calculate a multi-dimensional histogram of a dataset
///
/// # Parameters
///
/// * `sample` - Array of shape (N, D) containing N samples in D dimensions
/// * `bins` - Number of bins for each dimension. Can be:
///   - A single usize: Same number of bins for all dimensions
///   - A vector of usize: Different number of bins for each dimension
/// * `range` - Optional vector of (min, max) tuples for each dimension
/// * `weights` - Optional array of weights for each sample
///
/// # Returns
///
/// A tuple of (histogram counts, vector of bin edges for each dimension)
///
/// # Examples
///
/// ```
/// use numrs2::prelude::*;
///
/// // Create 2D data points
/// let data = Array::from_vec(vec![
///     0.0, 0.0,
///     0.5, 0.5,
///     1.0, 1.0,
///     0.3, 0.7,
/// ]).reshape(&[4, 2]);
///
/// // Compute 2D histogram with 2 bins in each dimension
/// let (hist, edges) = histogram_dd(&data, &[2, 2], None, None).expect("histogram_dd should succeed");
/// assert_eq!(hist.shape(), vec![2, 2]);
/// ```
pub fn histogram_dd<T: Float + Clone + NumCast + std::fmt::Display>(
    sample: &Array<T>,
    bins: &[usize],
    range: Option<Vec<(T, T)>>,
    weights: Option<&Array<T>>,
) -> Result<(Array<T>, Vec<Array<T>>)> {
    let shape = sample.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::InvalidOperation(
            "histogram_dd requires 2D input array of shape (N, D)".to_string(),
        ));
    }

    let n_samples = shape[0];
    let n_dims = shape[1];

    if n_samples == 0 || n_dims == 0 {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute histogram of empty data".to_string(),
        ));
    }

    // Validate bins
    if bins.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "bins array cannot be empty".to_string(),
        ));
    }

    let bin_counts = if bins.len() == 1 {
        // Use same number of bins for all dimensions
        vec![bins[0]; n_dims]
    } else if bins.len() == n_dims {
        bins.to_vec()
    } else {
        return Err(NumRs2Error::InvalidOperation(format!(
            "bins length {} does not match number of dimensions {}",
            bins.len(),
            n_dims
        )));
    };

    // Check for zero bins
    for &b in &bin_counts {
        if b == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Number of bins must be greater than 0".to_string(),
            ));
        }
    }

    // Validate weights if provided
    if let Some(w) = weights {
        if w.shape()[0] != n_samples {
            return Err(NumRs2Error::ShapeMismatch {
                expected: vec![n_samples],
                actual: w.shape().to_vec(),
            });
        }
    }

    // Determine ranges for each dimension
    let mut ranges = Vec::with_capacity(n_dims);
    let sample_data = sample.to_vec();

    if let Some(r) = range {
        if r.len() != n_dims {
            return Err(NumRs2Error::InvalidOperation(format!(
                "range length {} does not match number of dimensions {}",
                r.len(),
                n_dims
            )));
        }
        ranges = r;
    } else {
        // Compute min and max for each dimension
        for d in 0..n_dims {
            let mut min_val = sample_data[d];
            let mut max_val = sample_data[d];

            for i in 0..n_samples {
                let val = sample_data[i * n_dims + d];
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }

            // Add small epsilon to max to ensure last value is included
            let epsilon = T::from(1e-10).expect("epsilon should be representable");
            max_val = max_val + epsilon;

            ranges.push((min_val, max_val));
        }
    }

    // Create bin edges for each dimension
    let mut edges = Vec::with_capacity(n_dims);
    let mut bin_steps = Vec::with_capacity(n_dims);

    for (d, &n_bins) in bin_counts.iter().enumerate() {
        let (min_val, max_val) = ranges[d];
        let step = (max_val - min_val) / T::from(n_bins).expect("n_bins should be representable");
        bin_steps.push(step);

        let mut dim_edges = Vec::with_capacity(n_bins + 1);
        for i in 0..=n_bins {
            dim_edges.push(min_val + step * T::from(i).expect("index should be representable"));
        }
        edges.push(Array::from_vec(dim_edges));
    }

    // Initialize multi-dimensional histogram
    let hist_shape: Vec<usize> = bin_counts.clone();
    let total_bins: usize = hist_shape.iter().product();
    let mut hist_data = vec![T::zero(); total_bins];

    // Helper function to convert multi-dimensional indices to linear index
    let indices_to_linear = |indices: &[usize]| -> usize {
        let mut linear = 0;
        let mut stride = 1;
        for i in (0..n_dims).rev() {
            linear += indices[i] * stride;
            stride *= hist_shape[i];
        }
        linear
    };

    // Fill the histogram
    if let Some(w) = weights {
        let weights_data = w.to_vec();

        for i in 0..n_samples {
            let mut indices = Vec::with_capacity(n_dims);
            let mut in_bounds = true;

            for d in 0..n_dims {
                let val = sample_data[i * n_dims + d];
                let (min_val, max_val) = ranges[d];

                if val < min_val || val > max_val {
                    in_bounds = false;
                    break;
                }

                let mut idx = ((val - min_val) / bin_steps[d])
                    .to_usize()
                    .expect("bin index should be convertible");
                // Handle edge case where value equals max
                if idx >= bin_counts[d] {
                    idx = bin_counts[d] - 1;
                }
                indices.push(idx);
            }

            if in_bounds {
                let linear_idx = indices_to_linear(&indices);
                hist_data[linear_idx] = hist_data[linear_idx] + weights_data[i];
            }
        }
    } else {
        // No weights, just count
        for i in 0..n_samples {
            let mut indices = Vec::with_capacity(n_dims);
            let mut in_bounds = true;

            for d in 0..n_dims {
                let val = sample_data[i * n_dims + d];
                let (min_val, max_val) = ranges[d];

                if val < min_val || val > max_val {
                    in_bounds = false;
                    break;
                }

                let mut idx = ((val - min_val) / bin_steps[d])
                    .to_usize()
                    .expect("bin index should be convertible");
                // Handle edge case where value equals max
                if idx >= bin_counts[d] {
                    idx = bin_counts[d] - 1;
                }
                indices.push(idx);
            }

            if in_bounds {
                let linear_idx = indices_to_linear(&indices);
                hist_data[linear_idx] = hist_data[linear_idx] + T::one();
            }
        }
    }

    // Create the histogram array with proper shape
    let hist = Array::from_vec_shape(hist_data, &hist_shape)?;

    Ok((hist, edges))
}

/// Calculate counts of each unique value in an array
///
/// # Parameters
///
/// * `a` - Input array
/// * `weights` - Optional weights for each value
/// * `minlength` - Minimum length of the output array
///
/// # Returns
///
/// An array of counts for each value (assuming values are integers from 0 to n-1)
pub fn bincount<T: Float + Clone + NumCast + Send + Sync>(
    a: &Array<T>,
    weights: Option<&Array<T>>,
    minlength: Option<usize>,
) -> Result<Array<T>> {
    let data = a.to_vec();

    if data.is_empty() {
        let min_len = minlength.unwrap_or(0);
        let counts = vec![T::zero(); min_len];
        return Ok(Array::from_vec(counts));
    }

    // Find the maximum value to determine the output array size (no clone needed)
    let max_val = data
        .iter()
        .fold(data[0], |max, &val| if val > max { val } else { max });
    if max_val < T::zero() {
        return Err(NumRs2Error::InvalidOperation(
            "All values in bincount input array must be non-negative".to_string(),
        ));
    }

    let max_idx = max_val
        .to_usize()
        .expect("max value should be convertible to usize");
    let min_length = minlength.unwrap_or(0);
    let bin_count = (max_idx + 1).max(min_length);

    // Use parallel processing for large datasets
    let counts = if data.len() >= PARALLEL_THRESHOLD {
        let chunk_size = (64 * 1024 / std::mem::size_of::<T>())
            .max(1024)
            .min(data.len());

        if let Some(w) = weights {
            let weights_data = w.to_vec();

            if weights_data.len() != data.len() {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![data.len()],
                    actual: vec![weights_data.len()],
                });
            }

            // Check for negative values first
            if data.par_iter().any(|&val| val < T::zero()) {
                return Err(NumRs2Error::InvalidOperation(
                    "All values in bincount input array must be non-negative".to_string(),
                ));
            }

            // Parallel chunked bincount with weights
            let indices: Vec<usize> = (0..data.len()).collect();
            let partial_counts: Vec<Vec<T>> = indices
                .par_chunks(chunk_size)
                .map(|idx_chunk| {
                    let mut local_counts = vec![T::zero(); bin_count];
                    for &i in idx_chunk {
                        let idx = data[i]
                            .to_usize()
                            .expect("index should be convertible to usize");
                        if idx < bin_count {
                            local_counts[idx] = local_counts[idx] + weights_data[i];
                        }
                    }
                    local_counts
                })
                .collect();

            // Reduce partial counts
            let mut result = vec![T::zero(); bin_count];
            for partial in partial_counts {
                for (i, &count) in partial.iter().enumerate() {
                    result[i] = result[i] + count;
                }
            }
            result
        } else {
            // Check for negative values first
            if data.par_iter().any(|&val| val < T::zero()) {
                return Err(NumRs2Error::InvalidOperation(
                    "All values in bincount input array must be non-negative".to_string(),
                ));
            }

            // Parallel chunked bincount without weights
            let partial_counts: Vec<Vec<T>> = data
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_counts = vec![T::zero(); bin_count];
                    for &val in chunk {
                        let idx = val
                            .to_usize()
                            .expect("index should be convertible to usize");
                        if idx < bin_count {
                            local_counts[idx] = local_counts[idx] + T::one();
                        }
                    }
                    local_counts
                })
                .collect();

            // Reduce partial counts
            let mut result = vec![T::zero(); bin_count];
            for partial in partial_counts {
                for (i, &count) in partial.iter().enumerate() {
                    result[i] = result[i] + count;
                }
            }
            result
        }
    } else {
        // Sequential processing for small datasets
        let mut counts = vec![T::zero(); bin_count];

        if let Some(w) = weights {
            let weights_data = w.to_vec();

            if weights_data.len() != data.len() {
                return Err(NumRs2Error::ShapeMismatch {
                    expected: vec![data.len()],
                    actual: vec![weights_data.len()],
                });
            }

            for (i, &val) in data.iter().enumerate() {
                if val < T::zero() {
                    return Err(NumRs2Error::InvalidOperation(
                        "All values in bincount input array must be non-negative".to_string(),
                    ));
                }

                let idx = val
                    .to_usize()
                    .expect("index should be convertible to usize");
                if idx < bin_count {
                    counts[idx] = counts[idx] + weights_data[i];
                }
            }
        } else {
            for &val in &data {
                if val < T::zero() {
                    return Err(NumRs2Error::InvalidOperation(
                        "All values in bincount input array must be non-negative".to_string(),
                    ));
                }

                let idx = val
                    .to_usize()
                    .expect("index should be convertible to usize");
                if idx < bin_count {
                    counts[idx] = counts[idx] + T::one();
                }
            }
        }
        counts
    };

    Ok(Array::from_vec(counts))
}

/// Return the indices of the bins to which each value in input array belongs.
///
/// # Parameters
///
/// * `x` - Input array
/// * `bins` - Array of bin edges
/// * `right` - Whether the intervals include the right or the left bin edge
///
/// # Returns
///
/// Array of indices the same shape as x
pub fn digitize<T: Float + Clone + NumCast + Send + Sync>(
    x: &Array<T>,
    bins: &Array<T>,
    right: Option<bool>,
) -> Result<Array<usize>> {
    let x_data = x.to_vec();
    let bins_data = bins.to_vec();

    if bins_data.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Bins array cannot be empty".to_string(),
        ));
    }

    // Check if bins are monotonic
    let mut increasing = true;
    let mut decreasing = true;

    for i in 1..bins_data.len() {
        if bins_data[i] > bins_data[i - 1] {
            decreasing = false;
        }
        if bins_data[i] < bins_data[i - 1] {
            increasing = false;
        }
    }

    if !increasing && !decreasing {
        return Err(NumRs2Error::InvalidOperation(
            "Bins must be monotonically increasing or decreasing".to_string(),
        ));
    }

    // Determine bin membership
    let use_right = right.unwrap_or(false);
    let mut result = Vec::with_capacity(x_data.len());

    if increasing {
        for &val in &x_data {
            let mut idx = 0;
            for (i, &edge) in bins_data.iter().enumerate() {
                if (use_right && val <= edge) || (!use_right && val < edge) {
                    idx = i;
                    break;
                }
                // If we reach the last bin, index is equal to the number of bins
                idx = bins_data.len();
            }
            result.push(idx);
        }
    } else {
        // Bins are decreasing
        for &val in &x_data {
            let mut idx = 0;
            for (i, &edge) in bins_data.iter().enumerate() {
                if (use_right && val >= edge) || (!use_right && val > edge) {
                    idx = i;
                    break;
                }
                // If we reach the last bin, index is equal to the number of bins
                idx = bins_data.len();
            }
            result.push(idx);
        }
    }

    Ok(Array::from_vec(result))
}
