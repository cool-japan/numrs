//! Quantile and percentile functions
//!
//! This module provides quantile and percentile calculation following the
//! Hyndman & Fan (1996) taxonomy exactly as implemented by NumPy >= 1.22
//! (`numpy.quantile`/`numpy.percentile`, parameter `method=`):
//!
//! - quantile: Compute quantiles of a dataset
//! - percentile: Compute percentiles of a dataset
//!
//! ## Supported `method` values
//!
//! Continuous (Hyndman & Fan) methods, selected by `(alpha, beta)`:
//! * `"inverted_cdf"` (H&F type 1, discrete)
//! * `"averaged_inverted_cdf"` (H&F type 2, discrete with averaging)
//! * `"closest_observation"` (H&F type 3, discrete)
//! * `"interpolated_inverted_cdf"` (H&F type 4: alpha=0, beta=1)
//! * `"hazen"` (H&F type 5: alpha=0.5, beta=0.5)
//! * `"weibull"` (H&F type 6: alpha=0, beta=0)
//! * `"linear"` (H&F type 7: alpha=1, beta=1) -- **the default**, matching
//!   `numpy.quantile`'s default `method="linear"`.
//! * `"median_unbiased"` (H&F type 8: alpha=1/3, beta=1/3)
//! * `"normal_unbiased"` (H&F type 9: alpha=3/8, beta=3/8)
//!
//! Legacy (pre-NumPy-1.22) methods, kept for backward compatibility:
//! * `"lower"`, `"higher"`, `"nearest"`, `"midpoint"`
//!
//! All index/interpolation arithmetic is performed in `f64` regardless of the
//! array's element type `T`, matching NumPy's own float64-based computation
//! and avoiding generic-float rounding pitfalls; only the final interpolated
//! value is cast back to `T`.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, NumCast};

/// All method names accepted by [`quantile`] / [`percentile`].
const VALID_METHODS: &[&str] = &[
    "linear",
    "lower",
    "higher",
    "nearest",
    "midpoint",
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "median_unbiased",
    "normal_unbiased",
];

/// Round to the nearest integer, breaking exact ties towards the nearest
/// *even* integer (IEEE 754 "round half to even" / banker's rounding).
///
/// This matches `numpy.around`/`numpy.rint`, which NumPy's `"nearest"`
/// quantile method relies on -- unlike `f64::round`, which breaks ties away
/// from zero.
fn round_half_even_f64(x: f64) -> f64 {
    let floor_v = x.floor();
    let diff = x - floor_v;
    if diff < 0.5 {
        floor_v
    } else if diff > 0.5 {
        floor_v + 1.0
    } else {
        // Exactly halfway: round to even.
        let floor_i = floor_v as i64;
        if floor_i.rem_euclid(2) == 0 {
            floor_v
        } else {
            floor_v + 1.0
        }
    }
}

/// NumPy's `_lerp`: a numerically-symmetric linear interpolation between `a`
/// (at `t=0`) and `b` (at `t=1`). Using `b - (b-a)*(1-t)` for `t >= 0.5`
/// (instead of always `a + (b-a)*t`) keeps the result exact at `t=1` and
/// reduces floating point error near the upper end, matching NumPy bit for
/// bit in common cases.
fn symmetric_lerp(a: f64, b: f64, t: f64) -> f64 {
    if t >= 0.5 {
        b - (b - a) * (1.0 - t)
    } else {
        a + (b - a) * t
    }
}

/// Hyndman & Fan's generic virtual-index formula for continuous methods
/// (H&F types 4-9), parameterized by `(alpha, beta)`.
fn compute_virtual_index(n_f: f64, q: f64, alpha: f64, beta: f64) -> f64 {
    n_f * q + (alpha + q * (1.0 - alpha - beta)) - 1.0
}

/// Resolve a (possibly out-of-range) floating virtual index into a valid
/// `(previous_index, next_index, gamma)` triple, mirroring NumPy's
/// `_get_indexes` / `_get_gamma`.
///
/// When the virtual index falls outside `[0, n-1]`, `previous` and `next`
/// both collapse to the nearest valid boundary (`0` or `n-1`) with
/// `gamma = 0`, which makes the final interpolation return that boundary
/// value exactly (since `previous == next` there).
fn clamp_virtual_index(virtual_index: f64, n: usize) -> (usize, usize, f64) {
    let last = n - 1;
    if virtual_index < 0.0 {
        (0, 0, 0.0)
    } else if virtual_index >= last as f64 {
        (last, last, 0.0)
    } else {
        let previous = virtual_index.floor();
        (
            previous as usize,
            previous as usize + 1,
            virtual_index - previous,
        )
    }
}

/// NumPy's `_discrete_interpolation_to_boundaries`: resolves a (possibly
/// negative/fractional) index to `floor(index)` or `floor(index) + 1`
/// depending on a method-specific condition over the fractional part,
/// then clips the result to `[0, last]`. Used by `inverted_cdf` and
/// `closest_observation`, which pick an actual observation rather than
/// interpolating.
fn discrete_interpolation_to_boundary(
    index: f64,
    last: usize,
    use_previous: impl Fn(f64, f64) -> bool,
) -> usize {
    let previous = index.floor();
    let gamma = index - previous;
    let chosen = if use_previous(gamma, index) {
        previous
    } else {
        previous + 1.0
    };
    let clipped = chosen.max(0.0);
    (clipped as usize).min(last)
}

/// Resolve `(previous_index, next_index, gamma)` for a single quantile `q`
/// (already validated to be in `[0, 1]`) against a method name (already
/// validated to be one of [`VALID_METHODS`]).
fn resolve_indices(n: usize, method: &str, q: f64) -> (usize, usize, f64) {
    let n_f = n as f64;
    let last = n - 1;
    match method {
        "lower" => {
            let (i, _, _) = clamp_virtual_index(((n_f - 1.0) * q).floor(), n);
            (i, i, 0.0)
        }
        "higher" => {
            // `ceil` is only used to pick the index; clamp handles bounds.
            let idx = ((n_f - 1.0) * q).ceil();
            let i = (idx.max(0.0) as usize).min(last);
            (i, i, 0.0)
        }
        "nearest" => {
            let idx = round_half_even_f64((n_f - 1.0) * q);
            let i = (idx.max(0.0) as usize).min(last);
            (i, i, 0.0)
        }
        "midpoint" => {
            let lo = ((n_f - 1.0) * q).floor();
            let hi = ((n_f - 1.0) * q).ceil();
            clamp_virtual_index(0.5 * (lo + hi), n)
        }
        "inverted_cdf" => {
            let i =
                discrete_interpolation_to_boundary(n_f * q - 1.0, last, |gamma, _| gamma == 0.0);
            (i, i, 0.0)
        }
        "closest_observation" => {
            // "choose the nearest even order statistic at gamma=0"
            // (H&F 1996, pp. 362); order is 1-based, so 0-based indices use
            // the opposite (odd) parity check.
            let i = discrete_interpolation_to_boundary(n_f * q - 1.5, last, |gamma, index| {
                gamma == 0.0 && (index.floor() as i64).rem_euclid(2) == 1
            });
            (i, i, 0.0)
        }
        "averaged_inverted_cdf" => {
            let virtual_index = n_f * q - 1.0;
            let (previous, next, gamma) = clamp_virtual_index(virtual_index, n);
            if previous == next {
                // Clamped at a boundary: gamma is irrelevant (previous == next).
                (previous, next, gamma)
            } else {
                let raw_gamma = virtual_index - virtual_index.floor();
                let fixed_gamma = if raw_gamma == 0.0 { 0.5 } else { 1.0 };
                (previous, next, fixed_gamma)
            }
        }
        "interpolated_inverted_cdf" => {
            clamp_virtual_index(compute_virtual_index(n_f, q, 0.0, 1.0), n)
        }
        "hazen" => clamp_virtual_index(compute_virtual_index(n_f, q, 0.5, 0.5), n),
        "weibull" => clamp_virtual_index(compute_virtual_index(n_f, q, 0.0, 0.0), n),
        "median_unbiased" => {
            clamp_virtual_index(compute_virtual_index(n_f, q, 1.0 / 3.0, 1.0 / 3.0), n)
        }
        "normal_unbiased" => {
            clamp_virtual_index(compute_virtual_index(n_f, q, 3.0 / 8.0, 3.0 / 8.0), n)
        }
        // "linear" and any already-validated method fall through here;
        // `(n-1) * q` is preferred over `compute_virtual_index(n, q, 1, 1)`
        // to avoid rounding issues, matching NumPy's own comment.
        _ => clamp_virtual_index((n_f - 1.0) * q, n),
    }
}

/// Order two floats so that `NaN` sorts to the end, matching NumPy's
/// behavior (partition/sort treat `NaN` as the maximum).
fn cmp_nan_last<T: Float>(a: &T, b: &T) -> std::cmp::Ordering {
    match a.partial_cmp(b) {
        Some(ordering) => ordering,
        None => {
            if a.is_nan() && b.is_nan() {
                std::cmp::Ordering::Equal
            } else if a.is_nan() {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }
    }
}

/// Compute the quantiles of a dataset.
///
/// # Parameters
///
/// * `a` - Input array
/// * `q` - Quantile or sequence of quantiles to compute, in range `[0, 1]`
/// * `method` - Interpolation method (default `"linear"`); see the [module
///   documentation](self) for the full list of supported values.
///
/// # Returns
///
/// Array of quantile values. If the input contains any `NaN`, every
/// returned quantile is `NaN` (matching NumPy). An out-of-range or `NaN`
/// `q`, an unrecognized `method`, or an empty input array are reported as
/// errors.
pub fn quantile<T: Float + Clone + NumCast + std::fmt::Display + Send + Sync>(
    a: &Array<T>,
    q: &Array<T>,
    method: Option<&str>,
) -> Result<Array<T>> {
    let method_str = method.unwrap_or("linear");
    if !VALID_METHODS.contains(&method_str) {
        return Err(NumRs2Error::InvalidOperation(format!(
            "Invalid method '{}'. Must be one of: {}",
            method_str,
            VALID_METHODS.join(", ")
        )));
    }

    let mut sorted_data = a.to_vec();
    if sorted_data.is_empty() {
        return Err(NumRs2Error::InvalidOperation(
            "Cannot compute quantiles of an empty array".to_string(),
        ));
    }
    sorted_data.sort_by(cmp_nan_last);
    let n = sorted_data.len();
    let has_nan = sorted_data[n - 1].is_nan();

    let q_data = q.to_vec();
    let mut result = Vec::with_capacity(q_data.len());

    for &q_val in &q_data {
        if q_val.is_nan() || q_val < T::zero() || q_val > T::one() {
            return Err(NumRs2Error::InvalidOperation(format!(
                "Quantile value {} out of bounds [0, 1]",
                q_val
            )));
        }

        if has_nan {
            result.push(T::nan());
            continue;
        }

        let q_f64 = q_val
            .to_f64()
            .expect("quantile value should convert to f64");
        let (previous, next, gamma) = resolve_indices(n, method_str, q_f64);
        let previous_val = sorted_data[previous]
            .to_f64()
            .expect("sorted value should convert to f64");
        let next_val = sorted_data[next]
            .to_f64()
            .expect("sorted value should convert to f64");
        let interpolated = symmetric_lerp(previous_val, next_val, gamma);
        result.push(T::from(interpolated).expect("interpolated value should convert back to T"));
    }

    Ok(Array::from_vec(result))
}

/// Compute the percentiles of a dataset
///
/// # Parameters
///
/// * `a` - Input array
/// * `q` - Percentile or sequence of percentiles to compute, in range [0, 100]
/// * `method` - Method to use for percentile calculation (same as quantile)
///
/// # Returns
///
/// Array of percentile values
pub fn percentile<T: Float + Clone + NumCast + std::fmt::Display + Send + Sync>(
    a: &Array<T>,
    q: &Array<T>,
    method: Option<&str>,
) -> Result<Array<T>> {
    // Convert percentiles to quantiles (0-100 to 0-1)
    let quantiles = q.map(|x| x / T::from(100.0).expect("100.0 should be representable"));

    // Call quantile with the converted values
    quantile(a, &quantiles, method)
}
