//! WebAssembly bindings for NumRS2 statistical operations
//!
//! This module provides JavaScript-friendly wrappers for NumRS2's statistical functionality.
//! All operations use scirs2-stats for implementation following SCIRS2 policy.

use super::array::WasmArray;
use crate::array::Array;
use crate::stats::{corrcoef, cov, histogram, percentile, Statistics};
use wasm_bindgen::prelude::*;

/// Compute the mean of array elements
///
/// # Parameters
/// - `arr`: Input array
///
/// # Returns
/// Mean value
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 3, 4, 5], [5]);
/// console.log(mean(arr)); // 3.0
/// ```
#[wasm_bindgen]
pub fn mean(arr: &WasmArray) -> f64 {
    arr.mean()
}

/// Compute the median of array elements
///
/// # Parameters
/// - `arr`: Input array
///
/// # Returns
/// Median value
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 3, 4, 5], [5]);
/// console.log(median(arr)); // 3.0
/// ```
#[wasm_bindgen]
pub fn median(arr: &WasmArray) -> f64 {
    arr.percentile(0.5)
}

/// Compute the variance of array elements
///
/// # Parameters
/// - `arr`: Input array
///
/// # Returns
/// Variance value
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 3, 4, 5], [5]);
/// console.log(variance(arr)); // 2.0
/// ```
#[wasm_bindgen]
pub fn variance(arr: &WasmArray) -> f64 {
    arr.var()
}

/// Compute the standard deviation of array elements
///
/// # Parameters
/// - `arr`: Input array
///
/// # Returns
/// Standard deviation value
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 3, 4, 5], [5]);
/// console.log(std_dev(arr)); // ~1.414
/// ```
#[wasm_bindgen]
pub fn std_dev(arr: &WasmArray) -> f64 {
    arr.std()
}

/// Compute the minimum value in array
///
/// # Parameters
/// - `arr`: Input array
///
/// # Returns
/// Minimum value
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([3, 1, 4, 1, 5], [5]);
/// console.log(minimum(arr)); // 1.0
/// ```
#[wasm_bindgen]
pub fn minimum(arr: &WasmArray) -> f64 {
    arr.min()
}

/// Compute the maximum value in array
///
/// # Parameters
/// - `arr`: Input array
///
/// # Returns
/// Maximum value
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([3, 1, 4, 1, 5], [5]);
/// console.log(maximum(arr)); // 5.0
/// ```
#[wasm_bindgen]
pub fn maximum(arr: &WasmArray) -> f64 {
    arr.max()
}

/// Compute a percentile of array elements
///
/// # Parameters
/// - `arr`: Input array
/// - `q`: Percentile to compute (0.0 to 1.0)
///
/// # Returns
/// Result containing percentile value or error
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 3, 4, 5], [5]);
/// console.log(compute_percentile(arr, 0.25)); // 2.0 (25th percentile)
/// console.log(compute_percentile(arr, 0.75)); // 4.0 (75th percentile)
/// ```
#[wasm_bindgen]
pub fn compute_percentile(arr: &WasmArray, q: f64) -> Result<f64, JsValue> {
    if !(0.0..=1.0).contains(&q) {
        return Err(JsValue::from_str("Percentile must be between 0.0 and 1.0"));
    }

    Ok(arr.percentile(q))
}

/// Compute histogram of array data
///
/// # Parameters
/// - `arr`: Input array
/// - `bins`: Number of bins
///
/// # Returns
/// Result containing tuple of (counts, bin_edges) or error
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 2, 3, 3, 3, 4, 4, 5], [9]);
/// const [counts, bins] = compute_histogram(arr, 5);
/// ```
#[wasm_bindgen]
pub fn compute_histogram(arr: &WasmArray, bins: usize) -> Result<HistogramResult, JsValue> {
    if bins == 0 {
        return Err(JsValue::from_str("Number of bins must be greater than 0"));
    }

    let arr_vec = arr.to_vec();
    let arr_shape = arr.shape();
    let inner = Array::from_vec_shape(arr_vec, &arr_shape)?;

    histogram(&inner, bins, None, None, None)
        .map(|(counts, bin_edges)| HistogramResult {
            counts: WasmArray::from_array(counts),
            bin_edges: WasmArray::from_array(bin_edges),
        })
        .map_err(|e| JsValue::from_str(&format!("Histogram computation error: {}", e)))
}

/// Result type for histogram computation
#[wasm_bindgen]
pub struct HistogramResult {
    counts: WasmArray,
    bin_edges: WasmArray,
}

#[wasm_bindgen]
impl HistogramResult {
    /// Get the bin counts
    #[wasm_bindgen(getter)]
    pub fn counts(&self) -> WasmArray {
        // `WasmArray` clones are O(1) (Arc-backed copy-on-write `Array`),
        // so this avoids an unnecessary `to_vec()` + reconstruction round-trip.
        self.counts.clone()
    }

    /// Get the bin edges
    #[wasm_bindgen(getter)]
    pub fn bin_edges(&self) -> WasmArray {
        self.bin_edges.clone()
    }
}

/// Compute correlation coefficient between two arrays
///
/// # Parameters
/// - `x`: First array
/// - `y`: Second array (optional, if None computes correlation matrix)
///
/// # Returns
/// Result containing correlation coefficient(s) or error
///
/// # Example
/// ```javascript
/// const x = WasmArray.from_vec([1, 2, 3, 4, 5], [5]);
/// const y = WasmArray.from_vec([2, 4, 6, 8, 10], [5]);
/// const corr = correlation(x, y); // Should be close to 1.0
/// ```
#[wasm_bindgen]
pub fn correlation(x: &WasmArray, y: Option<WasmArray>) -> Result<WasmArray, JsValue> {
    let x_vec = x.to_vec();
    let x_shape = x.shape();
    let x_inner = Array::from_vec_shape(x_vec, &x_shape)?;

    let y_inner = y
        .as_ref()
        .map(|y_arr| {
            let y_vec = y_arr.to_vec();
            let y_shape = y_arr.shape();
            Array::from_vec_shape(y_vec, &y_shape)
        })
        .transpose()?;

    corrcoef(&x_inner, y_inner.as_ref(), None)
        .map(WasmArray::from_array)
        .map_err(|e| JsValue::from_str(&format!("Correlation computation error: {}", e)))
}

/// Compute covariance between two arrays
///
/// # Parameters
/// - `x`: First array
/// - `y`: Second array (optional, if None computes covariance matrix)
///
/// # Returns
/// Result containing covariance value(s) or error
///
/// # Example
/// ```javascript
/// const x = WasmArray.from_vec([1, 2, 3, 4, 5], [5]);
/// const y = WasmArray.from_vec([2, 4, 6, 8, 10], [5]);
/// const cov_val = covariance(x, y);
/// ```
#[wasm_bindgen]
pub fn covariance(x: &WasmArray, y: Option<WasmArray>) -> Result<WasmArray, JsValue> {
    let x_vec = x.to_vec();
    let x_shape = x.shape();
    let x_inner = Array::from_vec_shape(x_vec, &x_shape)?;

    let y_inner = y
        .as_ref()
        .map(|y_arr| {
            let y_vec = y_arr.to_vec();
            let y_shape = y_arr.shape();
            Array::from_vec_shape(y_vec, &y_shape)
        })
        .transpose()?;

    cov(&x_inner, y_inner.as_ref(), None, None, None)
        .map(WasmArray::from_array)
        .map_err(|e| JsValue::from_str(&format!("Covariance computation error: {}", e)))
}

/// Compute sum of array elements
///
/// # Parameters
/// - `arr`: Input array
///
/// # Returns
/// Sum of all elements
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 3, 4, 5], [5]);
/// console.log(sum(arr)); // 15.0
/// ```
#[wasm_bindgen]
pub fn sum(arr: &WasmArray) -> f64 {
    arr.sum()
}

/// Compute product of array elements
///
/// # Parameters
/// - `arr`: Input array
///
/// # Returns
/// Product of all elements
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 3, 4, 5], [5]);
/// console.log(product(arr)); // 120.0
/// ```
#[wasm_bindgen]
pub fn product(arr: &WasmArray) -> f64 {
    let arr_vec = arr.to_vec();
    arr_vec.iter().product()
}

// Statistical helper trait implementation for WasmArray
impl WasmArray {
    /// Internal helper to get percentile
    pub(crate) fn percentile(&self, q: f64) -> f64 {
        self.inner().percentile(q)
    }

    /// Internal helper to get variance
    pub(crate) fn var(&self) -> f64 {
        let m = self.mean();
        let arr_vec = self.to_vec();
        let sum_sq_diff: f64 = arr_vec.iter().map(|&x| (x - m).powi(2)).sum();
        sum_sq_diff / (arr_vec.len() as f64)
    }

    /// Internal helper to get standard deviation
    pub(crate) fn std(&self) -> f64 {
        self.var().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let arr =
            WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("from_vec should succeed");
        assert_eq!(mean(&arr), 3.0);
    }

    #[test]
    fn test_median() {
        let arr =
            WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("from_vec should succeed");
        assert_eq!(median(&arr), 3.0);
    }

    #[test]
    fn test_variance() {
        let arr =
            WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("from_vec should succeed");
        let var = variance(&arr);
        assert!((var - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let arr =
            WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("from_vec should succeed");
        let std = std_dev(&arr);
        assert!((std - 1.4142135623730951).abs() < 1e-10);
    }

    #[test]
    fn test_min_max() {
        let arr =
            WasmArray::from_vec(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]).expect("from_vec should succeed");
        assert_eq!(minimum(&arr), 1.0);
        assert_eq!(maximum(&arr), 5.0);
    }

    #[test]
    fn test_percentile() {
        let arr =
            WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("from_vec should succeed");
        let p25 = compute_percentile(&arr, 0.25).expect("percentile should succeed");
        let p75 = compute_percentile(&arr, 0.75).expect("percentile should succeed");
        assert!((1.0..=3.0).contains(&p25));
        assert!((3.0..=5.0).contains(&p75));
    }

    // `WasmArray::percentile` (the `pub(crate)` helper behind `median()` and
    // `compute_percentile()`) reads through `WasmArray::inner()` -- a plain
    // borrow of the wrapped `Array<f64>` -- rather than the previous
    // `to_vec()` + `Array::from_vec_shape()` round-trip. That is only a
    // no-op simplification if reading a *non-contiguous* (e.g. transposed)
    // `Array` in logical order still visits every element exactly once,
    // with the same values, as reading an already-flat array with the same
    // logical layout. These two tests pin that invariant down directly, so
    // a future change to `Array::to_vec()`, `WasmArray::inner()`, or
    // `percentile()` that broke it would fail here rather than silently
    // shipping a wrong median/percentile to JS callers.
    #[test]
    fn test_transpose_preserves_logical_order_for_to_vec() {
        // [[1, 2, 3],      [[1, 4],
        //  [4, 5, 6]]  -T->  [2, 5],
        //                    [3, 6]]
        // Row-major flatten of the transpose is [1, 4, 2, 5, 3, 6], not the
        // untouched backing buffer [1, 2, 3, 4, 5, 6].
        let matrix = WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("from_vec should succeed");
        let transposed = matrix.transpose();
        assert_eq!(transposed.shape(), vec![3, 2]);
        assert_eq!(transposed.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_percentile_after_transpose_matches_equivalent_flat_array() {
        let matrix = WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .expect("from_vec should succeed");
        let transposed = matrix.transpose();
        // Same six values, same logical order as `transposed.to_vec()`
        // (see test above), laid out contiguously from the start.
        let flat = WasmArray::from_vec(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[6])
            .expect("from_vec should succeed");

        for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let expected = call_percentile(&flat, q);
            let actual = call_percentile(&transposed, q);
            assert!(
                (actual - expected).abs() < 1e-12,
                "percentile({q}) mismatch after transpose: expected {expected}, got {actual}"
            );
        }
    }

    // `WasmArray::percentile` is `pub(crate)`, not `#[wasm_bindgen]`-exported,
    // so it is directly callable from this same-crate test module.
    fn call_percentile(arr: &WasmArray, q: f64) -> f64 {
        arr.percentile(q)
    }

    #[test]
    fn test_sum_product() {
        let arr =
            WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("from_vec should succeed");
        assert_eq!(sum(&arr), 15.0);
        assert_eq!(product(&arr), 120.0);
    }
}
