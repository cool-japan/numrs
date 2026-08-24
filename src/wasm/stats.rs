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

    histogram(&inner, bins, None, None)
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
        let counts_vec = self.counts.to_vec();
        let counts_shape = self.counts.shape();
        WasmArray::from_array(Array::from_vec_shape(counts_vec, &counts_shape)?)
    }

    /// Get the bin edges
    #[wasm_bindgen(getter)]
    pub fn bin_edges(&self) -> WasmArray {
        let edges_vec = self.bin_edges.to_vec();
        let edges_shape = self.bin_edges.shape();
        WasmArray::from_array(Array::from_vec_shape(edges_vec, &edges_shape)?)
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

    let y_inner = y.as_ref().map(|y_arr| {
        let y_vec = y_arr.to_vec();
        let y_shape = y_arr.shape();
        Array::from_vec_shape(y_vec, &y_shape)?
    });

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

    let y_inner = y.as_ref().map(|y_arr| {
        let y_vec = y_arr.to_vec();
        let y_shape = y_arr.shape();
        Array::from_vec_shape(y_vec, &y_shape)?
    });

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
        let arr_vec = self.to_vec();
        let arr_shape = self.shape();
        let inner = Array::from_vec_shape(arr_vec, &arr_shape)?;

        inner.percentile(q)
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

    #[test]
    fn test_sum_product() {
        let arr =
            WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("from_vec should succeed");
        assert_eq!(sum(&arr), 15.0);
        assert_eq!(product(&arr), 120.0);
    }
}
