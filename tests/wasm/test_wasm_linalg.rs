//! WebAssembly linear algebra tests
//!
//! This module tests NumRS2's WASM linear algebra bindings using wasm-bindgen-test.
//! All operations use scirs2-linalg (pure Rust via OxiBLAS) following SCIRS2 policy.

#![cfg(target_arch = "wasm32")]

// `wasm_bindgen_test_configure!(run_in_browser)` must appear exactly once per
// test binary (see `tests/wasm_integration.rs`, which wires this file in),
// not once per module -- so it does not appear here.
use wasm_bindgen_test::*;

// Import WASM bindings
use numrs2::wasm::linalg::*;
use numrs2::wasm::WasmArray;

#[wasm_bindgen_test]
fn test_matmul_basic() {
    // 2x3 @ 3x2 = 2x2
    let a = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("Failed to create array");

    let b = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2])
        .expect("Failed to create array");

    let c = matmul(&a, &b).expect("Matrix multiplication failed");

    assert_eq!(c.shape(), vec![2, 2]);

    // Verify result values
    assert_eq!(c.get(&[0, 0]).expect("Get failed"), 22.0);
    assert_eq!(c.get(&[0, 1]).expect("Get failed"), 28.0);
    assert_eq!(c.get(&[1, 0]).expect("Get failed"), 49.0);
    assert_eq!(c.get(&[1, 1]).expect("Get failed"), 64.0);
}

#[wasm_bindgen_test]
fn test_matmul_identity() {
    let a =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Failed to create array");

    let identity =
        WasmArray::from_vec(&vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).expect("Failed to create identity");

    let c = matmul(&a, &identity).expect("Matrix multiplication failed");

    // A @ I should equal A
    assert_eq!(c.get(&[0, 0]).expect("Get failed"), 1.0);
    assert_eq!(c.get(&[0, 1]).expect("Get failed"), 2.0);
    assert_eq!(c.get(&[1, 0]).expect("Get failed"), 3.0);
    assert_eq!(c.get(&[1, 1]).expect("Get failed"), 4.0);
}

#[wasm_bindgen_test]
fn test_matmul_incompatible_shapes() {
    let a = WasmArray::ones(&[2, 3]);
    let b = WasmArray::ones(&[2, 3]);

    // 2x3 @ 2x3 should fail
    let result = matmul(&a, &b);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_matmul_non_2d() {
    let a = WasmArray::ones(&[5]);
    let b = WasmArray::ones(&[5]);

    // 1D arrays should fail
    let result = matmul(&a, &b);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_dot_product() {
    let a = WasmArray::from_vec(&vec![1.0, 2.0, 3.0], &[3]).expect("Failed to create array");

    let b = WasmArray::from_vec(&vec![4.0, 5.0, 6.0], &[3]).expect("Failed to create array");

    let result = dot_product(&a, &b).expect("Dot product failed");

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(result, 32.0);
}

#[wasm_bindgen_test]
fn test_dot_product_orthogonal() {
    let a = WasmArray::from_vec(&vec![1.0, 0.0], &[2]).expect("Failed to create array");

    let b = WasmArray::from_vec(&vec![0.0, 1.0], &[2]).expect("Failed to create array");

    let result = dot_product(&a, &b).expect("Dot product failed");

    // Orthogonal vectors should have dot product of 0
    assert_eq!(result, 0.0);
}

#[wasm_bindgen_test]
fn test_dot_product_incompatible_lengths() {
    let a = WasmArray::ones(&[3]);
    let b = WasmArray::ones(&[4]);

    // Different lengths should fail
    let result = dot_product(&a, &b);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_dot_product_non_1d() {
    let a = WasmArray::ones(&[2, 2]);
    let b = WasmArray::ones(&[2, 2]);

    // 2D arrays should fail
    let result = dot_product(&a, &b);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_outer_product() {
    let a = WasmArray::from_vec(&vec![1.0, 2.0], &[2]).expect("Failed to create array");

    let b = WasmArray::from_vec(&vec![3.0, 4.0, 5.0], &[3]).expect("Failed to create array");

    let result = outer_product(&a, &b).expect("Outer product failed");

    assert_eq!(result.shape(), vec![2, 3]);

    // [[1*3, 1*4, 1*5],
    //  [2*3, 2*4, 2*5]]
    assert_eq!(result.get(&[0, 0]).expect("Get failed"), 3.0);
    assert_eq!(result.get(&[0, 1]).expect("Get failed"), 4.0);
    assert_eq!(result.get(&[0, 2]).expect("Get failed"), 5.0);
    assert_eq!(result.get(&[1, 0]).expect("Get failed"), 6.0);
    assert_eq!(result.get(&[1, 1]).expect("Get failed"), 8.0);
    assert_eq!(result.get(&[1, 2]).expect("Get failed"), 10.0);
}

#[wasm_bindgen_test]
fn test_outer_product_non_1d() {
    let a = WasmArray::ones(&[2, 2]);
    let b = WasmArray::ones(&[3]);

    // 2D array should fail
    let result = outer_product(&a, &b);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_norm_l2() {
    // 3-4-5 right triangle
    let arr = WasmArray::from_vec(&vec![3.0, 4.0], &[2]).expect("Failed to create array");

    let norm = compute_norm(&arr, 2.0).expect("Norm computation failed");

    assert_eq!(norm, 5.0);
}

#[wasm_bindgen_test]
fn test_norm_l1() {
    let arr = WasmArray::from_vec(&vec![3.0, 4.0], &[2]).expect("Failed to create array");

    let norm = compute_norm(&arr, 1.0).expect("Norm computation failed");

    // L1 norm: |3| + |4| = 7
    assert_eq!(norm, 7.0);
}

#[wasm_bindgen_test]
fn test_norm_infinity() {
    let arr = WasmArray::from_vec(&vec![3.0, -5.0, 2.0], &[3]).expect("Failed to create array");

    let norm = compute_norm(&arr, f64::INFINITY).expect("Norm computation failed");

    // Infinity norm: max(|3|, |-5|, |2|) = 5
    assert_eq!(norm, 5.0);
}

#[wasm_bindgen_test]
fn test_trace() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Failed to create array");

    let tr = trace(&arr).expect("Trace computation failed");

    // Trace: 1 + 4 = 5
    assert_eq!(tr, 5.0);
}

#[wasm_bindgen_test]
fn test_trace_3x3() {
    let arr = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
        .expect("Failed to create array");

    let tr = trace(&arr).expect("Trace computation failed");

    // Trace: 1 + 5 + 9 = 15
    assert_eq!(tr, 15.0);
}

#[wasm_bindgen_test]
fn test_trace_non_square() {
    let arr = WasmArray::ones(&[2, 3]);

    // Non-square matrix should fail
    let result = trace(&arr);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_trace_non_2d() {
    let arr = WasmArray::ones(&[5]);

    // 1D array should fail
    let result = trace(&arr);
    assert!(result.is_err());
}

// LAPACK-dependent tests (only run when lapack feature is enabled)

#[cfg(feature = "lapack")]
#[wasm_bindgen_test]
fn test_determinant() {
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Failed to create array");

    let det = determinant(&arr).expect("Determinant computation failed");

    // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
    assert!((det - (-2.0)).abs() < 1e-10);
}

#[cfg(feature = "lapack")]
#[wasm_bindgen_test]
fn test_determinant_identity() {
    let identity =
        WasmArray::from_vec(&vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).expect("Failed to create array");

    let det = determinant(&identity).expect("Determinant computation failed");

    // Identity matrix has determinant 1
    assert!((det - 1.0).abs() < 1e-10);
}

#[cfg(feature = "lapack")]
#[wasm_bindgen_test]
fn test_determinant_singular() {
    // Singular matrix (linearly dependent rows)
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 2.0, 4.0], &[2, 2]).expect("Failed to create array");

    let det = determinant(&arr).expect("Determinant computation failed");

    // Singular matrix has determinant 0
    assert!(det.abs() < 1e-10);
}

#[cfg(feature = "lapack")]
#[wasm_bindgen_test]
fn test_determinant_non_square() {
    let arr = WasmArray::ones(&[2, 3]);

    let result = determinant(&arr);
    assert!(result.is_err());
}

#[cfg(feature = "lapack")]
#[wasm_bindgen_test]
fn test_matrix_inverse() {
    let arr =
        WasmArray::from_vec(&vec![4.0, 7.0, 2.0, 6.0], &[2, 2]).expect("Failed to create array");

    // The wasm binding is named `inverse`, not `matrix_inverse`.
    let inv = inverse(&arr).expect("Matrix inversion failed");

    assert_eq!(inv.shape(), vec![2, 2]);

    // Verify A @ A^(-1) = I
    let product = matmul(&arr, &inv).expect("Matrix multiplication failed");

    let tolerance = 1e-10;
    assert!((product.get(&[0, 0]).expect("Get failed") - 1.0).abs() < tolerance);
    assert!(product.get(&[0, 1]).expect("Get failed").abs() < tolerance);
    assert!(product.get(&[1, 0]).expect("Get failed").abs() < tolerance);
    assert!((product.get(&[1, 1]).expect("Get failed") - 1.0).abs() < tolerance);
}

#[cfg(feature = "lapack")]
#[wasm_bindgen_test]
fn test_matrix_inverse_singular() {
    // Singular matrix cannot be inverted
    let arr =
        WasmArray::from_vec(&vec![1.0, 2.0, 2.0, 4.0], &[2, 2]).expect("Failed to create array");

    let result = inverse(&arr);
    assert!(result.is_err());
}

#[cfg(feature = "lapack")]
#[wasm_bindgen_test]
fn test_matrix_inverse_non_square() {
    let arr = WasmArray::ones(&[2, 3]);

    let result = inverse(&arr);
    assert!(result.is_err());
}

// `test_svd`, `test_qr_decomposition`, `test_eigenvalues` and
// `test_eigenvalues_non_square` were removed here (W6-TESTS orphaned-test
// pass): they called `singular_value_decomposition`/`qr_decomposition`/
// `compute_eigenvalues`, none of which exist in `numrs2::wasm::linalg` --
// that module only exposes `matmul`/`dot_product`/`outer_product`/
// `compute_norm`/`trace`/`determinant`/`inverse` (grep confirms no SVD/QR/
// eigenvalue binding anywhere under `src/wasm/`). This is a real gap in the
// WASM binding surface (the module doc-comment at `tests/wasm/mod.rs`
// describes SVD/QR/eigenvalue coverage that was apparently planned but never
// implemented), not a test bug -- worth a follow-up task to add those
// bindings to `src/wasm/linalg.rs`, which is out of this pass's file
// ownership (tests/** only).

#[wasm_bindgen_test]
fn test_transpose_in_linalg_context() {
    let a = WasmArray::from_vec(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("Failed to create array");

    let at = a.transpose();

    assert_eq!(at.shape(), vec![3, 2]);

    // Verify transpose correctness
    assert_eq!(
        a.get(&[0, 1]).expect("Get failed"),
        at.get(&[1, 0]).expect("Get failed")
    );
    assert_eq!(
        a.get(&[1, 2]).expect("Get failed"),
        at.get(&[2, 1]).expect("Get failed")
    );
}

#[wasm_bindgen_test]
fn test_matrix_chain_operations() {
    // Test A @ B @ C
    let a = WasmArray::ones(&[2, 3]);
    let b = WasmArray::ones(&[3, 4]);
    let c = WasmArray::ones(&[4, 2]);

    let ab = matmul(&a, &b).expect("First matmul failed");
    let abc = matmul(&ab, &c).expect("Second matmul failed");

    assert_eq!(abc.shape(), vec![2, 2]);
}
