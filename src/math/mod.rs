//! Mathematical functions for NumRS2 arrays
//!
//! This module provides comprehensive mathematical operations for arrays,
//! organized into logical submodules:
//!
//! - `elementwise` - SIMD-optimized element-wise operations (sin, cos, exp, log, etc.)
//! - `creation` - Array creation functions (zeros, ones, linspace, meshgrid, etc.)
//! - `diff` - Differentiation and integration (diff, trapz)
//! - `statistics` - Statistical functions (mean, std, var, cumsum, etc.)
//! - `aggregation` - Aggregation functions (sum, max, min, sort)
//! - `nan_handling` - NaN-aware functions (nansum, nanmean, nanstd, etc.)
//! - `search` - Search and binning (searchsorted, digitize, bincount)
//! - `checking` - Value checking (isnan, isinf, isfinite)
//! - `utilities` - Utility functions (gcd, lcm, copysign, etc.)
//! - `window` - Window functions (hanning, hamming, blackman, kaiser)
//! - `gradient` - Gradient and additional math functions
//! - `nan_extrema` - NaN-aware extrema functions
//!
//! # Example
//!
//! ```
//! use numrs2::prelude::*;
//! use numrs2::math::{linspace, mean, std};
//!
//! let x = linspace(0.0, 1.0, 100);
//! let mean_val = mean(&x, None, false).expect("mean failed");
//! let std_val = std(&x, None, 0, false).expect("std failed");
//! ```

// Submodules
mod aggregation;
mod checking;
mod creation;
mod diff;
mod elementwise;
mod gradient;
mod nan_extrema;
mod nan_handling;
mod search;
mod statistics;
mod utilities;
mod window;

// Re-export helper functions for use by submodules
pub(crate) use elementwise::{from_nd_array, to_nd_array_f32, to_nd_array_f64};

// Re-export everything from submodules

// Element-wise math trait and implementation
pub use elementwise::ElementWiseMath;

// Array creation functions
pub use creation::{
    angle, arange, complex_abs, conj, empty, geomspace, imag, linspace, logspace, meshgrid,
    meshgrid2d, mgrid, ogrid, ones, real, unwrap, zeros,
};

// Differentiation and integration
pub use diff::{diff, diff_extended, ediff1d, trapz};

// Statistics functions
pub use statistics::{
    argmax, argmin, argsort, around, clip, cumprod, cumsum, kurtosis, mean, prod, resize, skew,
    std, var,
};

// Aggregation functions
pub use aggregation::{
    amax, amin, argpartition, cumulative_prod, cumulative_sum, max, min, round, sort, sum,
};

// NaN-handling functions
pub use nan_handling::{
    nancumsum, nanmax, nanmean, nanmin, nanpercentile, nanprod, nanquantile, nanstd, nansum, nanvar,
};

// Search and binning functions
pub use search::{bincount, digitize, interp, median, partition, searchsorted};

// Value checking functions
pub use checking::{
    count_nonzero, flatnonzero, iscomplex, isfinite, isinf, isnan, isneginf, isnormal, isposinf,
    isreal, nan_to_num, nonzero,
};

// Utility functions
pub use utilities::{
    copysign, divmod, fmod, frexp, gcd, heaviside, lcm, ldexp, modf, nextafter, real_if_close,
    remainder, sinc,
};

// Window functions
pub use window::{bartlett, blackman, hamming, hanning, i0, kaiser};

// Gradient and additional math functions
pub use gradient::{
    fix, fmax, fmin, gradient, negative, positive, reciprocal, rint, signbit, GradientSpacing,
};

// NaN-aware extrema functions
pub use nan_extrema::{nanargmax, nanargmin, nancumprod};
