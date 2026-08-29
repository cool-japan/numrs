//! WebAssembly bindings for NumRS2 linear algebra operations
//!
//! This module provides JavaScript-friendly wrappers for NumRS2's linear algebra functionality.
//! All operations use scirs2-linalg (pure Rust via OxiBLAS) following SCIRS2 policy.

use super::array::WasmArray;
use crate::array::Array;
use crate::linalg::{norm, outer, vdot};
use wasm_bindgen::prelude::*;

/// Matrix multiplication
///
/// Computes the matrix product of two arrays.
///
/// # Parameters
/// - `a`: First matrix
/// - `b`: Second matrix
///
/// # Returns
/// Result containing product matrix or error
///
/// # Example
/// ```javascript
/// const a = WasmArray.ones([2, 3]);
/// const b = WasmArray.ones([3, 2]);
/// const c = matmul(a, b);
/// console.log(c.shape()); // [2, 2]
/// ```
#[wasm_bindgen]
pub fn matmul(a: &WasmArray, b: &WasmArray) -> Result<WasmArray, JsValue> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.is_empty() || b_shape.is_empty() {
        return Err(JsValue::from_str("Arrays must have at least 1 dimension"));
    }

    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(JsValue::from_str(
            "Matrix multiplication requires 2D arrays",
        ));
    }

    if a_shape[1] != b_shape[0] {
        return Err(JsValue::from_str(&format!(
            "Incompatible shapes for matrix multiplication: [{}, {}] and [{}, {}]",
            a_shape[0], a_shape[1], b_shape[0], b_shape[1]
        )));
    }

    // Extract inner arrays and perform matmul
    let a_inner = unsafe { &*(a as *const WasmArray as *const Array<f64>) };
    let b_inner = unsafe { &*(b as *const WasmArray as *const Array<f64>) };

    a_inner
        .matmul(b_inner)
        .map(WasmArray::from_array)
        .map_err(|e| JsValue::from_str(&format!("Matrix multiplication error: {}", e)))
}

/// Compute the dot product of two vectors
///
/// # Parameters
/// - `a`: First vector
/// - `b`: Second vector
///
/// # Returns
/// Result containing dot product or error
///
/// # Example
/// ```javascript
/// const a = WasmArray.from_vec([1, 2, 3], [3]);
/// const b = WasmArray.from_vec([4, 5, 6], [3]);
/// const result = dot_product(a, b); // 32.0
/// ```
#[wasm_bindgen]
pub fn dot_product(a: &WasmArray, b: &WasmArray) -> Result<f64, JsValue> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 1 || b_shape.len() != 1 {
        return Err(JsValue::from_str(
            "Dot product requires 1D arrays (vectors)",
        ));
    }

    if a_shape[0] != b_shape[0] {
        return Err(JsValue::from_str(&format!(
            "Vectors must have the same length: {} vs {}",
            a_shape[0], b_shape[0]
        )));
    }

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    let a_array = Array::from_vec(a_vec);
    let b_array = Array::from_vec(b_vec);

    vdot(&a_array, &b_array).map_err(|e| JsValue::from_str(&format!("Dot product error: {}", e)))
}

/// Compute the outer product of two vectors
///
/// # Parameters
/// - `a`: First vector
/// - `b`: Second vector
///
/// # Returns
/// Result containing outer product matrix or error
///
/// # Example
/// ```javascript
/// const a = WasmArray.from_vec([1, 2], [2]);
/// const b = WasmArray.from_vec([3, 4, 5], [3]);
/// const result = outer_product(a, b);
/// console.log(result.shape()); // [2, 3]
/// ```
#[wasm_bindgen]
pub fn outer_product(a: &WasmArray, b: &WasmArray) -> Result<WasmArray, JsValue> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 1 || b_shape.len() != 1 {
        return Err(JsValue::from_str(
            "Outer product requires 1D arrays (vectors)",
        ));
    }

    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    let a_array = Array::from_vec(a_vec);
    let b_array = Array::from_vec(b_vec);

    outer(&a_array, &b_array)
        .map(WasmArray::from_array)
        .map_err(|e| JsValue::from_str(&format!("Outer product error: {}", e)))
}

/// Compute the norm of a vector or matrix
///
/// # Parameters
/// - `arr`: Input array
/// - `ord`: Order of the norm (1, 2, or Infinity)
///
/// # Returns
/// Result containing norm value or error
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([3, 4], [2]);
/// const l2_norm = compute_norm(arr, 2.0); // 5.0
/// ```
#[wasm_bindgen]
pub fn compute_norm(arr: &WasmArray, ord: f64) -> Result<f64, JsValue> {
    let arr_vec = arr.to_vec();
    let arr_shape = arr.shape();
    let inner = Array::from_vec_shape(arr_vec, &arr_shape)?;

    let ord_option = if ord.is_finite() {
        Some(ord)
    } else {
        Some(f64::INFINITY)
    };

    norm(&inner, ord_option)
        .map_err(|e| JsValue::from_str(&format!("Norm computation error: {}", e)))
}

/// Compute matrix trace (sum of diagonal elements)
///
/// # Parameters
/// - `arr`: Input square matrix
///
/// # Returns
/// Result containing trace or error
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 3, 4], [2, 2]);
/// const tr = trace(arr); // 5.0 (1 + 4)
/// ```
#[wasm_bindgen]
pub fn trace(arr: &WasmArray) -> Result<f64, JsValue> {
    let arr_shape = arr.shape();

    if arr_shape.len() != 2 {
        return Err(JsValue::from_str("Trace requires a 2D matrix"));
    }

    if arr_shape[0] != arr_shape[1] {
        return Err(JsValue::from_str("Trace requires a square matrix"));
    }

    let arr_vec = arr.to_vec();
    let inner = Array::from_vec_shape(arr_vec, &arr_shape)?;

    crate::linalg::trace(&inner)
        .map_err(|e| JsValue::from_str(&format!("Trace computation error: {}", e)))
}

// LAPACK-dependent operations (only available with lapack feature)

/// Compute matrix determinant
///
/// **Note:** This function is only available when NumRS2 is compiled with the `lapack` feature.
///
/// # Parameters
/// - `arr`: Input square matrix
///
/// # Returns
/// Result containing determinant or error
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 3, 4], [2, 2]);
/// const det = determinant(arr); // -2.0
/// ```
#[cfg(feature = "lapack")]
#[wasm_bindgen]
pub fn determinant(arr: &WasmArray) -> Result<f64, JsValue> {
    let arr_shape = arr.shape();

    if arr_shape.len() != 2 {
        return Err(JsValue::from_str("Determinant requires a 2D matrix"));
    }

    if arr_shape[0] != arr_shape[1] {
        return Err(JsValue::from_str("Determinant requires a square matrix"));
    }

    let arr_vec = arr.to_vec();
    let inner = Array::from_vec_shape(arr_vec, &arr_shape)?;

    crate::linalg::det(&inner)
        .map_err(|e| JsValue::from_str(&format!("Determinant computation error: {}", e)))
}

/// Compute matrix inverse
///
/// **Note:** This function is only available when NumRS2 is compiled with the `lapack` feature.
///
/// # Parameters
/// - `arr`: Input square matrix
///
/// # Returns
/// Result containing inverse matrix or error
///
/// # Example
/// ```javascript
/// const arr = WasmArray.from_vec([1, 2, 3, 4], [2, 2]);
/// const inv = inverse(arr);
/// ```
#[cfg(feature = "lapack")]
#[wasm_bindgen]
pub fn inverse(arr: &WasmArray) -> Result<WasmArray, JsValue> {
    let arr_shape = arr.shape();

    if arr_shape.len() != 2 {
        return Err(JsValue::from_str("Inverse requires a 2D matrix"));
    }

    if arr_shape[0] != arr_shape[1] {
        return Err(JsValue::from_str("Inverse requires a square matrix"));
    }

    let arr_vec = arr.to_vec();
    let inner = Array::from_vec_shape(arr_vec, &arr_shape)?;

    crate::linalg::inv(&inner)
        .map(WasmArray::from_array)
        .map_err(|e| JsValue::from_str(&format!("Matrix inversion error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() {
        let a = WasmArray::ones(&[2, 3]);
        let b = WasmArray::ones(&[3, 2]);
        let c = matmul(&a, &b).expect("matmul should succeed");
        assert_eq!(c.shape(), vec![2, 2]);
        assert_eq!(c.sum(), 12.0); // Each element is 3.0, 4 elements total
    }

    #[test]
    fn test_dot_product() {
        let a = WasmArray::from_vec(&[1.0, 2.0, 3.0], &[3]).expect("from_vec should succeed");
        let b = WasmArray::from_vec(&[4.0, 5.0, 6.0], &[3]).expect("from_vec should succeed");
        let result = dot_product(&a, &b).expect("dot_product should succeed");
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_norm() {
        let arr = WasmArray::from_vec(&[3.0, 4.0], &[2]).expect("from_vec should succeed");
        let l2_norm = compute_norm(&arr, 2.0).expect("norm should succeed");
        assert!((l2_norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_trace() {
        let arr =
            WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("from_vec should succeed");
        let tr = trace(&arr).expect("trace should succeed");
        assert_eq!(tr, 5.0); // 1 + 4
    }

    #[cfg(feature = "lapack")]
    #[test]
    fn test_determinant() {
        let arr =
            WasmArray::from_vec(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("from_vec should succeed");
        let det = determinant(&arr).expect("determinant should succeed");
        assert!((det - (-2.0)).abs() < 1e-10);
    }
}
