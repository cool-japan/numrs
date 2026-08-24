//! Statistical functions for NumRS2
//!
//! This module provides various statistical operations including:
//! - Basic statistics (mean, var, std, min, max, percentile)
//! - Correlation and covariance (cov, corrcoef)
//! - Quantiles and percentiles
//! - Histograms (1D, 2D, and N-dimensional)
//! - NaN-aware statistics (nanmean, nanstd, nanvar, etc.)
//! - Mode calculation
//! - Distribution functions (PDF, CDF, quantile functions)

pub mod basic;
pub mod correlation;
pub mod distributions;
pub mod histogram;
pub mod mode;
pub mod nan_stats;
pub mod quantile;

// Re-export Statistics trait and PARALLEL_THRESHOLD constant
pub use basic::{
    average, average_with_weights, max_along_axis, min_along_axis, ptp, Statistics,
    PARALLEL_THRESHOLD,
};

// Re-export correlation functions
pub use correlation::{corrcoef, cov};

// Re-export quantile functions
pub use quantile::{percentile, quantile};

// Re-export histogram functions and types
pub use histogram::{
    bincount, digitize, histogram, histogram2d, histogram_bin_edges, histogram_dd, histogramdd,
    BinSpec, HistBins,
};

// Re-export NaN-aware statistics
pub use nan_stats::{nanmax, nanmean, nanmin, nanprod, nanstd, nansum, nanvar};

// Re-export mode function
pub use mode::mode;

// Re-export distribution functions
pub use distributions::{
    beta_cdf, beta_logpdf, beta_pdf, beta_ppf, cauchy_cdf, cauchy_pdf, cauchy_ppf, gamma_cdf,
    gamma_logpdf, gamma_pdf, gamma_ppf, laplace_cdf, laplace_pdf, laplace_ppf, logistic_cdf,
    logistic_pdf, logistic_ppf, pareto_cdf, pareto_pdf, pareto_ppf, student_t_cdf, student_t_pdf,
    student_t_ppf,
};
