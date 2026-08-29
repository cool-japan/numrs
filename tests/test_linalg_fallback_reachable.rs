//! Companion to `test_linalg_default_reachable.rs`: exercises the
//! complementary fallback implementations (pure-Rust, no LAPACK) that the
//! cfg gap fix made reachable for `matrix_decomp` ON / `lapack` OFF
//! (previously *neither* impl block compiled in this combination).
//!
//! This test only compiles when the primary `all(matrix_decomp, lapack)`
//! path is *not* active, i.e. it covers the historical default feature set
//! (`matrix_decomp` ON, `lapack` OFF) among other combinations.

#![cfg(not(all(feature = "matrix_decomp", feature = "lapack")))]

use numrs2::prelude::*;

const TOL: f64 = 1e-9;

/// Same matrix as the primary-path smoke test, det = 9 (see that file for
/// the hand-worked cofactor expansion).
fn sample_matrix() -> Array<f64> {
    Array::from_vec(vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 2.0]).reshape(&[3, 3])
}

#[test]
fn fallback_det_matches_known_value() {
    let a = sample_matrix();
    let det = a
        .det()
        .expect("fallback det() should be reachable and succeed");
    assert!((det - 9.0).abs() < TOL, "expected det ≈ 9.0, got {det}");
}

#[test]
fn fallback_inv_round_trips_to_identity() {
    let a = sample_matrix();
    let inv = a
        .inv()
        .expect("fallback inv() should be reachable and succeed");
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
fn fallback_solve_recovers_known_solution() {
    let a = sample_matrix();
    let x_expected = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let x_col = x_expected.reshape(&[3, 1]);
    let b_col = a.matmul(&x_col).expect("matmul should succeed");
    let b = Array::from_vec(b_col.to_vec());

    let x = a
        .solve(&b)
        .expect("fallback solve() should be reachable and succeed");
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
fn fallback_svd_cholesky_qr_never_panic() {
    // These methods must always return a `Result` (Ok when lapack happens
    // to be available alongside a disabled matrix_decomp, honest
    // FeatureNotEnabled otherwise) rather than failing to compile or
    // panicking, for every combination this file's cfg admits.
    let a = sample_matrix();
    let _ = a.svd();
    let _ = a.cholesky();
    let _ = a.qr();

    // Without lapack specifically, svd must be the honest error.
    if !cfg!(feature = "lapack") {
        assert!(
            a.svd().is_err(),
            "svd() without lapack should return an error, not panic"
        );
    }
}
