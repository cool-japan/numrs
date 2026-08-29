//! WebAssembly bindings for NumRS2 Array operations
//!
//! This module provides JavaScript-friendly wrappers for NumRS2's core Array type.

use crate::array::Array;
use crate::stats::Statistics;
use wasm_bindgen::prelude::*;

/// WebAssembly wrapper for NumRS2 Array
///
/// This struct provides JavaScript-friendly bindings for NumRS2's Array type.
/// All operations return Results and avoid unwrap() calls for robust error handling.
///
/// `Array<f64>` is `Arc`-backed (copy-on-write), so deriving `Clone` here is
/// an O(1) reference-count bump, not a deep copy.
#[derive(Clone)]
#[wasm_bindgen]
pub struct WasmArray {
    inner: Array<f64>,
}

#[wasm_bindgen]
impl WasmArray {
    /// Create a new array filled with zeros
    ///
    /// # Parameters
    /// - `shape`: Array shape as a JavaScript array of numbers
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.zeros([2, 3]);
    /// console.log(arr.shape()); // [2, 3]
    /// ```
    #[wasm_bindgen]
    pub fn zeros(shape: &[usize]) -> WasmArray {
        WasmArray {
            inner: Array::zeros(shape),
        }
    }

    /// Create a new array filled with ones
    ///
    /// # Parameters
    /// - `shape`: Array shape as a JavaScript array of numbers
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.ones([2, 3]);
    /// ```
    #[wasm_bindgen]
    pub fn ones(shape: &[usize]) -> WasmArray {
        WasmArray {
            inner: Array::ones(shape),
        }
    }

    /// Create a new array filled with a constant value
    ///
    /// # Parameters
    /// - `shape`: Array shape as a JavaScript array of numbers
    /// - `value`: Fill value
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.full([2, 3], 5.0);
    /// ```
    #[wasm_bindgen]
    pub fn full(shape: &[usize], value: f64) -> WasmArray {
        WasmArray {
            inner: Array::full(shape, value),
        }
    }

    /// Create array from a flat JavaScript array with shape
    ///
    /// # Parameters
    /// - `data`: Flat array of values
    /// - `shape`: Array shape
    ///
    /// # Returns
    /// Result containing WasmArray or error message
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.from_vec([1, 2, 3, 4, 5, 6], [2, 3]);
    /// ```
    #[wasm_bindgen]
    pub fn from_vec(data: &[f64], shape: &[usize]) -> Result<WasmArray, JsValue> {
        let total_size: usize = shape.iter().product();
        if data.len() != total_size {
            return Err(JsValue::from_str(&format!(
                "Data length {} does not match shape product {}",
                data.len(),
                total_size
            )));
        }

        Ok(WasmArray {
            inner: Array::from_vec_shape(data.to_vec(), shape)?,
        })
    }

    /// Get the shape of the array
    ///
    /// # Returns
    /// JavaScript array containing the shape dimensions
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.zeros([2, 3]);
    /// console.log(arr.shape()); // [2, 3]
    /// ```
    #[wasm_bindgen]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape()
    }

    /// Get the number of dimensions
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.zeros([2, 3]);
    /// console.log(arr.ndim()); // 2
    /// ```
    #[wasm_bindgen]
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Get the total number of elements
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.zeros([2, 3]);
    /// console.log(arr.size()); // 6
    /// ```
    #[wasm_bindgen]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    /// Reshape the array to a new shape
    ///
    /// # Parameters
    /// - `new_shape`: New shape as JavaScript array
    ///
    /// # Returns
    /// Result containing reshaped WasmArray or error
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.zeros([2, 3]);
    /// const reshaped = arr.reshape([3, 2]);
    /// ```
    #[wasm_bindgen]
    pub fn reshape(&self, new_shape: &[usize]) -> Result<WasmArray, JsValue> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.inner.size() {
            return Err(JsValue::from_str(&format!(
                "Cannot reshape array of size {} into shape with size {}",
                self.inner.size(),
                new_size
            )));
        }

        Ok(WasmArray {
            inner: self.inner.reshape(new_shape),
        })
    }

    /// Transpose the array
    ///
    /// # Returns
    /// New WasmArray with transposed dimensions
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.zeros([2, 3]);
    /// const t = arr.transpose();
    /// console.log(t.shape()); // [3, 2]
    /// ```
    #[wasm_bindgen]
    pub fn transpose(&self) -> WasmArray {
        WasmArray {
            inner: self.inner.transpose(),
        }
    }

    /// Get element at specified indices
    ///
    /// # Parameters
    /// - `indices`: Array indices as JavaScript array
    ///
    /// # Returns
    /// Result containing the value or error
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.full([2, 3], 5.0);
    /// const val = arr.get([0, 1]); // 5.0
    /// ```
    #[wasm_bindgen]
    pub fn get(&self, indices: &[usize]) -> Result<f64, JsValue> {
        self.inner
            .get(indices)
            .map_err(|e| JsValue::from_str(&format!("Get error: {}", e)))
    }

    /// Set element at specified indices
    ///
    /// # Parameters
    /// - `indices`: Array indices as JavaScript array
    /// - `value`: Value to set
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.zeros([2, 3]);
    /// arr.set([0, 1], 5.0);
    /// ```
    #[wasm_bindgen]
    pub fn set(&mut self, indices: &[usize], value: f64) -> Result<(), JsValue> {
        self.inner
            .set(indices, value)
            .map_err(|e| JsValue::from_str(&format!("Set error: {}", e)))
    }

    /// Convert array to flat JavaScript array
    ///
    /// # Returns
    /// Flat array of all elements in row-major order
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.full([2, 3], 5.0);
    /// const data = arr.to_vec(); // [5, 5, 5, 5, 5, 5]
    /// ```
    #[wasm_bindgen]
    pub fn to_vec(&self) -> Vec<f64> {
        self.inner.to_vec()
    }

    /// Element-wise addition
    ///
    /// # Parameters
    /// - `other`: Another WasmArray
    ///
    /// # Returns
    /// Result containing sum array or error
    ///
    /// # Example
    /// ```javascript
    /// const a = WasmArray.ones([2, 3]);
    /// const b = WasmArray.ones([2, 3]);
    /// const c = a.add(b);
    /// ```
    #[wasm_bindgen]
    pub fn add(&self, other: &WasmArray) -> Result<WasmArray, JsValue> {
        if self.inner.shape() != other.inner.shape() {
            return Err(JsValue::from_str("Arrays must have the same shape"));
        }

        Ok(WasmArray {
            inner: self.inner.add(&other.inner),
        })
    }

    /// Element-wise subtraction
    ///
    /// # Parameters
    /// - `other`: Another WasmArray
    ///
    /// # Returns
    /// Result containing difference array or error
    #[wasm_bindgen]
    pub fn subtract(&self, other: &WasmArray) -> Result<WasmArray, JsValue> {
        if self.inner.shape() != other.inner.shape() {
            return Err(JsValue::from_str("Arrays must have the same shape"));
        }

        Ok(WasmArray {
            inner: self.inner.subtract(&other.inner),
        })
    }

    /// Element-wise multiplication
    ///
    /// # Parameters
    /// - `other`: Another WasmArray
    ///
    /// # Returns
    /// Result containing product array or error
    #[wasm_bindgen]
    pub fn multiply(&self, other: &WasmArray) -> Result<WasmArray, JsValue> {
        if self.inner.shape() != other.inner.shape() {
            return Err(JsValue::from_str("Arrays must have the same shape"));
        }

        Ok(WasmArray {
            inner: self.inner.multiply(&other.inner),
        })
    }

    /// Element-wise division
    ///
    /// # Parameters
    /// - `other`: Another WasmArray
    ///
    /// # Returns
    /// Result containing quotient array or error
    #[wasm_bindgen]
    pub fn divide(&self, other: &WasmArray) -> Result<WasmArray, JsValue> {
        if self.inner.shape() != other.inner.shape() {
            return Err(JsValue::from_str("Arrays must have the same shape"));
        }

        Ok(WasmArray {
            inner: self.inner.divide(&other.inner),
        })
    }

    /// Add a scalar value to all elements
    ///
    /// # Parameters
    /// - `scalar`: Scalar value to add
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.ones([2, 3]);
    /// const result = arr.add_scalar(5.0);
    /// ```
    #[wasm_bindgen]
    pub fn add_scalar(&self, scalar: f64) -> WasmArray {
        WasmArray {
            inner: self.inner.add_scalar(scalar),
        }
    }

    /// Multiply all elements by a scalar value
    ///
    /// # Parameters
    /// - `scalar`: Scalar value to multiply by
    #[wasm_bindgen]
    pub fn multiply_scalar(&self, scalar: f64) -> WasmArray {
        WasmArray {
            inner: self.inner.multiply_scalar(scalar),
        }
    }

    /// Compute the sum of all elements
    ///
    /// # Returns
    /// Sum of all array elements
    ///
    /// # Example
    /// ```javascript
    /// const arr = WasmArray.full([2, 3], 5.0);
    /// console.log(arr.sum()); // 30.0
    /// ```
    #[wasm_bindgen]
    pub fn sum(&self) -> f64 {
        self.inner.sum()
    }

    /// Compute the mean of all elements
    ///
    /// # Returns
    /// Mean of all array elements
    #[wasm_bindgen]
    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Compute the minimum element value
    ///
    /// # Returns
    /// Minimum value in the array
    #[wasm_bindgen]
    pub fn min(&self) -> f64 {
        self.inner.min()
    }

    /// Compute the maximum element value
    ///
    /// # Returns
    /// Maximum value in the array
    #[wasm_bindgen]
    pub fn max(&self) -> f64 {
        self.inner.max()
    }
}

// Helper implementation for WasmArray - provides internal methods
impl WasmArray {
    /// Create a WasmArray from an Array
    ///
    /// # Parameters
    /// - `array`: The Array to wrap
    ///
    /// # Returns
    /// A new WasmArray wrapping the given Array
    pub(crate) fn from_array(array: Array<f64>) -> WasmArray {
        WasmArray { inner: array }
    }

    /// Borrow the inner Array
    ///
    /// # Returns
    /// A reference to the contained Array<f64>, avoiding an unnecessary
    /// `to_vec()` + reconstruction round-trip for callers in sibling
    /// modules that only need to read the data.
    pub(crate) fn inner(&self) -> &Array<f64> {
        &self.inner
    }

    /// Consume this WasmArray and return the inner Array
    ///
    /// # Returns
    /// The contained Array<f64>
    // Consuming counterpart to `inner()` (which is used); no sibling module
    // needs ownership yet, only a borrow, so this has no call sites today.
    #[allow(dead_code)]
    pub(crate) fn into_inner(self) -> Array<f64> {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let arr = WasmArray::zeros(&[2, 3]);
        assert_eq!(arr.shape(), vec![2, 3]);
        assert_eq!(arr.size(), 6);
    }

    #[test]
    fn test_ones() {
        let arr = WasmArray::ones(&[2, 3]);
        assert_eq!(arr.sum(), 6.0);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = WasmArray::from_vec(&data, &[2, 3]).expect("from_vec should succeed");
        assert_eq!(arr.shape(), vec![2, 3]);
        assert_eq!(arr.to_vec(), data);
    }

    #[test]
    fn test_reshape() {
        let arr = WasmArray::zeros(&[2, 3]);
        let reshaped = arr.reshape(&[3, 2]).expect("reshape should succeed");
        assert_eq!(reshaped.shape(), vec![3, 2]);
    }

    #[test]
    fn test_transpose() {
        let arr = WasmArray::zeros(&[2, 3]);
        let t = arr.transpose();
        assert_eq!(t.shape(), vec![3, 2]);
    }

    #[test]
    fn test_arithmetic() {
        let a = WasmArray::ones(&[2, 3]);
        let b = WasmArray::full(&[2, 3], 2.0);

        let sum = a.add(&b).expect("add should succeed");
        assert_eq!(sum.sum(), 18.0); // (1 + 2) * 6 = 18

        let diff = b.subtract(&a).expect("subtract should succeed");
        assert_eq!(diff.sum(), 6.0); // (2 - 1) * 6 = 6

        let prod = a.multiply(&b).expect("multiply should succeed");
        assert_eq!(prod.sum(), 12.0); // (1 * 2) * 6 = 12

        let quot = b.divide(&a).expect("divide should succeed");
        assert_eq!(quot.sum(), 12.0); // (2 / 1) * 6 = 12
    }

    #[test]
    fn test_scalar_ops() {
        let arr = WasmArray::ones(&[2, 3]);
        let added = arr.add_scalar(5.0);
        assert_eq!(added.sum(), 36.0); // (1 + 5) * 6 = 36

        let scaled = arr.multiply_scalar(3.0);
        assert_eq!(scaled.sum(), 18.0); // (1 * 3) * 6 = 18
    }
}
