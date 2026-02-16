//! WebAssembly array operations tests
//!
//! This module tests NumRS2's WASM array bindings using wasm-bindgen-test.
//! These tests run in a browser environment via wasm-pack test.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

// Import WASM bindings
use numrs2::wasm::WasmArray;

#[wasm_bindgen_test]
fn test_create_zeros_array() {
    let arr = WasmArray::zeros(&[2, 3]);

    let shape = arr.shape();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], 2);
    assert_eq!(shape[1], 3);
    assert_eq!(arr.size(), 6);
    assert_eq!(arr.ndim(), 2);
}

#[wasm_bindgen_test]
fn test_create_ones_array() {
    let arr = WasmArray::ones(&[3, 4]);

    let shape = arr.shape();
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], 3);
    assert_eq!(shape[1], 4);
    assert_eq!(arr.size(), 12);

    // Check first element
    let val = arr.get(&[0, 0]).expect("Failed to get element");
    assert_eq!(val, 1.0);
}

#[wasm_bindgen_test]
fn test_create_full_array() {
    let arr = WasmArray::full(&[2, 2], 5.0);

    assert_eq!(arr.size(), 4);

    let val = arr.get(&[1, 1]).expect("Failed to get element");
    assert_eq!(val, 5.0);
}

#[wasm_bindgen_test]
fn test_from_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = WasmArray::from_vec(&data, &[2, 3]).expect("Failed to create array from vec");

    assert_eq!(arr.shape(), vec![2, 3]);
    assert_eq!(arr.size(), 6);

    let val = arr.get(&[0, 0]).expect("Failed to get element");
    assert_eq!(val, 1.0);

    let val = arr.get(&[1, 2]).expect("Failed to get element");
    assert_eq!(val, 6.0);
}

#[wasm_bindgen_test]
fn test_from_vec_invalid_shape() {
    let data = vec![1.0, 2.0, 3.0];
    let result = WasmArray::from_vec(&data, &[2, 3]);

    // Should fail because 3 elements != 2*3 = 6 elements
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_shape_and_ndim() {
    let arr = WasmArray::zeros(&[2, 3, 4]);

    assert_eq!(arr.ndim(), 3);
    assert_eq!(arr.shape(), vec![2, 3, 4]);
    assert_eq!(arr.size(), 24);
}

#[wasm_bindgen_test]
fn test_reshape() {
    let arr = WasmArray::zeros(&[2, 3]);
    let reshaped = arr.reshape(&[3, 2]).expect("Failed to reshape");

    assert_eq!(reshaped.shape(), vec![3, 2]);
    assert_eq!(reshaped.size(), 6);
}

#[wasm_bindgen_test]
fn test_reshape_invalid() {
    let arr = WasmArray::zeros(&[2, 3]);
    let result = arr.reshape(&[2, 2]);

    // Should fail because 6 elements != 4 elements
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_transpose() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = WasmArray::from_vec(&data, &[2, 3]).expect("Failed to create array");

    let transposed = arr.transpose();
    assert_eq!(transposed.shape(), vec![3, 2]);

    // Check that transpose worked correctly
    let val = arr.get(&[0, 1]).expect("Failed to get element");
    let val_t = transposed.get(&[1, 0]).expect("Failed to get element");
    assert_eq!(val, val_t);
}

#[wasm_bindgen_test]
fn test_get_and_set() {
    let mut arr = WasmArray::zeros(&[3, 3]);

    // Set some values
    arr.set(&[0, 0], 1.0).expect("Failed to set");
    arr.set(&[1, 1], 2.0).expect("Failed to set");
    arr.set(&[2, 2], 3.0).expect("Failed to set");

    // Get values back
    assert_eq!(arr.get(&[0, 0]).expect("Failed to get"), 1.0);
    assert_eq!(arr.get(&[1, 1]).expect("Failed to get"), 2.0);
    assert_eq!(arr.get(&[2, 2]).expect("Failed to get"), 3.0);
    assert_eq!(arr.get(&[0, 1]).expect("Failed to get"), 0.0);
}

#[wasm_bindgen_test]
fn test_get_invalid_index() {
    let arr = WasmArray::zeros(&[2, 3]);

    // Out of bounds
    let result = arr.get(&[5, 5]);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_set_invalid_index() {
    let mut arr = WasmArray::zeros(&[2, 3]);

    // Out of bounds
    let result = arr.set(&[5, 5], 1.0);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_to_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let arr = WasmArray::from_vec(&data, &[2, 2]).expect("Failed to create array");

    let result = arr.to_vec();
    assert_eq!(result, data);
}

#[wasm_bindgen_test]
fn test_add_arrays() {
    let a = WasmArray::ones(&[2, 3]);
    let b = WasmArray::full(&[2, 3], 2.0);

    let c = a.add(&b).expect("Failed to add arrays");

    assert_eq!(c.shape(), vec![2, 3]);
    let val = c.get(&[0, 0]).expect("Failed to get");
    assert_eq!(val, 3.0);
}

#[wasm_bindgen_test]
fn test_add_incompatible_shapes() {
    let a = WasmArray::ones(&[2, 3]);
    let b = WasmArray::ones(&[3, 2]);

    let result = a.add(&b);
    // Should fail due to shape mismatch
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_subtract_arrays() {
    let a = WasmArray::full(&[2, 3], 5.0);
    let b = WasmArray::full(&[2, 3], 2.0);

    let c = a.subtract(&b).expect("Failed to subtract arrays");

    let val = c.get(&[0, 0]).expect("Failed to get");
    assert_eq!(val, 3.0);
}

#[wasm_bindgen_test]
fn test_multiply_arrays() {
    let a = WasmArray::full(&[2, 3], 3.0);
    let b = WasmArray::full(&[2, 3], 2.0);

    let c = a.multiply(&b).expect("Failed to multiply arrays");

    let val = c.get(&[0, 0]).expect("Failed to get");
    assert_eq!(val, 6.0);
}

#[wasm_bindgen_test]
fn test_divide_arrays() {
    let a = WasmArray::full(&[2, 3], 6.0);
    let b = WasmArray::full(&[2, 3], 2.0);

    let c = a.divide(&b).expect("Failed to divide arrays");

    let val = c.get(&[0, 0]).expect("Failed to get");
    assert_eq!(val, 3.0);
}

#[wasm_bindgen_test]
fn test_scalar_operations() {
    let arr = WasmArray::full(&[2, 3], 2.0);

    // Add scalar
    let result = arr.add_scalar(3.0);
    let val = result.get(&[0, 0]).expect("Failed to get");
    assert_eq!(val, 5.0);

    // Multiply scalar
    let result = arr.multiply_scalar(3.0);
    let val = result.get(&[0, 0]).expect("Failed to get");
    assert_eq!(val, 6.0);
}

#[wasm_bindgen_test]
fn test_sum() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = WasmArray::from_vec(&data, &[2, 3]).expect("Failed to create array");

    let sum = arr.sum();
    assert_eq!(sum, 21.0);
}

#[wasm_bindgen_test]
fn test_mean() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let arr = WasmArray::from_vec(&data, &[2, 3]).expect("Failed to create array");

    let mean = arr.mean();
    assert_eq!(mean, 3.5);
}

#[wasm_bindgen_test]
fn test_min_max() {
    let data = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0];
    let arr = WasmArray::from_vec(&data, &[2, 3]).expect("Failed to create array");

    assert_eq!(arr.min(), 1.0);
    assert_eq!(arr.max(), 9.0);
}

#[wasm_bindgen_test]
fn test_std() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let arr = WasmArray::from_vec(&data, &[5]).expect("Failed to create array");

    let std = arr.std();
    // Standard deviation should be around 1.414
    assert!((std - 1.414).abs() < 0.01);
}

#[wasm_bindgen_test]
fn test_matmul() {
    // 2x3 matrix
    let a = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("Failed to create array");

    // 3x2 matrix
    let b = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
        .expect("Failed to create array");

    let c = a.matmul(&b).expect("Failed to multiply matrices");

    assert_eq!(c.shape(), vec![2, 2]);

    // Result should be:
    // [[22, 28],
    //  [49, 64]]
    assert_eq!(c.get(&[0, 0]).expect("Failed to get"), 22.0);
    assert_eq!(c.get(&[0, 1]).expect("Failed to get"), 28.0);
    assert_eq!(c.get(&[1, 0]).expect("Failed to get"), 49.0);
    assert_eq!(c.get(&[1, 1]).expect("Failed to get"), 64.0);
}

#[wasm_bindgen_test]
fn test_matmul_incompatible() {
    let a = WasmArray::ones(&[2, 3]);
    let b = WasmArray::ones(&[2, 3]);

    // Should fail: 2x3 @ 2x3 is invalid
    let result = a.matmul(&b);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_clone() {
    let arr = WasmArray::full(&[2, 3], 5.0);
    let cloned = arr.clone_array();

    assert_eq!(arr.shape(), cloned.shape());
    assert_eq!(arr.to_vec(), cloned.to_vec());
}

#[wasm_bindgen_test]
fn test_1d_array() {
    let arr = WasmArray::zeros(&[10]);

    assert_eq!(arr.ndim(), 1);
    assert_eq!(arr.size(), 10);
    assert_eq!(arr.shape(), vec![10]);
}

#[wasm_bindgen_test]
fn test_3d_array() {
    let arr = WasmArray::zeros(&[2, 3, 4]);

    assert_eq!(arr.ndim(), 3);
    assert_eq!(arr.size(), 24);
    assert_eq!(arr.shape(), vec![2, 3, 4]);
}

#[wasm_bindgen_test]
fn test_empty_array() {
    let arr = WasmArray::zeros(&[0]);

    assert_eq!(arr.size(), 0);
    assert_eq!(arr.shape(), vec![0]);
}
