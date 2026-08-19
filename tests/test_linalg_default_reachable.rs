//! Regression test guarding against the `matrix_decomp` + `lapack` cfg gap.
//!
//! Historically, the core linalg API (`det`, `inv`, `solve`, `svd`, `eig`,
//! `cholesky`, `qr`, `lstsq`, `lu`, `pinv`, `matrix_rank`, `slogdet`) was
//! gated `#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]` with
//! fallback arms gated `#[cfg(not(feature = "matrix_decomp"))]`. With the
//! old default features (`matrix_decomp` ON, `lapack` OFF), *neither* arm
//! compiled, so these methods silently vanished from the default build.
//!
//! This test only compiles when both `matrix_decomp` and `lapack` are
//! enabled, and exercises the primary (LAPACK-backed) implementations to
//! confirm they are reachable and numerically correct.

#![cfg(all(feature = "matrix_decomp", feature = "lapack"))]

use numrs2::prelude::*;

const TOL: f64 = 1e-10;

/// A well-conditioned, non-symmetric, invertible 3x3 matrix with a hand
/// verifiable determinant:
///
/// | 2  1  1 |
/// | 1  3  2 |
/// | 1  0  2 |
///
/// det = 2*(3*2 - 2*0) - 1*(1*2 - 2*1) + 1*(1*0 - 3*1) = 12 - 0 - 3 = 9
fn sample_matrix() -> Array<f64> {
    Array::from_vec(vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0]).reshape(&[3, 3])
}

#[test]
fn det_matches_known_value() {
    let a = sample_matrix();
    let det = a.det().expect("det() should be reachable and succeed");
    assert!((det - 9.0).abs() < TOL, "expected det ≈ 9.0, got {det}");
}

#[test]
fn inv_round_trips_to_identity() {
    let a = sample_matrix();
    let inv = a.inv().expect("inv() should be reachable and succeed");
    let product = a.matmul(&inv).expect("matmul should succeed");

    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = product.get(&[i, j]).expect("in-bounds get");
            assert!(
                (actual - expected).abs() < TOL,
                "A * A^-1 [{i},{j}] = {actual}, expected {expected}"
            );
        }
    }
}

#[test]
fn svd_reconstructs_original_matrix() {
    let a = sample_matrix();
    let (u, s, vt) = a.svd().expect("svd() should be reachable and succeed");

    // Rebuild the diagonal singular-value matrix S (3x3) from the singular
    // value vector, then reconstruct A' = U * S * V^T.
    let k = s.shape()[0];
    let mut s_diag = Array::<f64>::zeros(&[3, 3]);
    for i in 0..k {
        let val = s.get(&[i]).expect("in-bounds get");
        s_diag.set(&[i, i], val).expect("in-bounds set");
    }

    let us = u.matmul(&s_diag).expect("matmul should succeed");
    let reconstructed = us.matmul(&vt).expect("matmul should succeed");

    for i in 0..3 {
        for j in 0..3 {
            let expected = a.get(&[i, j]).expect("in-bounds get");
            let actual = reconstructed.get(&[i, j]).expect("in-bounds get");
            assert!(
                (actual - expected).abs() < 1e-8,
                "SVD reconstruction [{i},{j}] = {actual}, expected {expected}"
            );
        }
    }
}

#[test]
fn solve_recovers_known_solution() {
    let a = sample_matrix();
    // Choose x = [1, 2, 3] and derive b = A * x, so the expected solution is known.
    let x_expected = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let x_col = x_expected.reshape(&[3, 1]);
    let b_col = a.matmul(&x_col).expect("matmul should succeed");
    let b = Array::from_vec(b_col.to_vec());

    let x = a
        .solve(&b)
        .expect("solve() should be reachable and succeed");
    let x_data = x.to_vec();
    let expected_data = [1.0, 2.0, 3.0];

    for (actual, expected) in x_data.iter().zip(expected_data.iter()) {
        assert!(
            (actual - expected).abs() < TOL,
            "solve() result {actual} does not match expected {expected}"
        );
    }
}

#[test]
fn free_function_solve_and_inv_are_reachable() {
    // These free functions (numrs2::linalg::{solve, inv}) delegate to the
    // inherent methods above, and are re-exported through the prelude.
    let a = sample_matrix();
    let b = Array::from_vec(vec![4.0, 8.0, 5.0]);

    let x = numrs2::linalg::solve(&a, &b).expect("free-function solve() should be reachable");
    let inv = numrs2::linalg::inv(&a).expect("free-function inv() should be reachable");

    // Sanity: A * solve(A, b) ≈ b
    let x_col = Array::from_vec(x.to_vec()).reshape(&[3, 1]);
    let b_reconstructed = a.matmul(&x_col).expect("matmul should succeed");
    for (i, expected) in b.to_vec().iter().enumerate() {
        let actual = b_reconstructed.get(&[i, 0]).expect("in-bounds get");
        assert!(
            (actual - expected).abs() < TOL,
            "A * x [{i}] = {actual}, expected {expected}"
        );
    }

    // Sanity: A * inv(A) ≈ I
    let product = a.matmul(&inv).expect("matmul should succeed");
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = product.get(&[i, j]).expect("in-bounds get");
            assert!(
                (actual - expected).abs() < TOL,
                "A * inv(A) [{i},{j}] = {actual}, expected {expected}"
            );
        }
    }
}
