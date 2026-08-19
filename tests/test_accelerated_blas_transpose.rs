//! Regression tests for `AcceleratedBlas::gemm` / `AcceleratedBlas::gemv`
//! transpose flag handling.
//!
//! # Background
//!
//! `AcceleratedBlas::gemm`/`gemv` used to accept `trans_a`/`trans_b`/`trans`
//! flags named `_trans_a`/`_trans_b`/`_trans` and never consult them — the
//! flags were silently ignored, so a caller asking for `Aᵀ·B` got `A·B`
//! instead. With non-square 2x3/3x2 operands this either surfaces as a
//! spurious dimension-mismatch error, or — for the `trans_a && trans_b`
//! combination, where the swapped-shape operands happen to still be
//! conformable to *some* product — as a silently wrong result of the wrong
//! shape. These tests pin the CORRECT, BLAS-standard semantics:
//!
//! `C = alpha * op(A) * op(B) + beta * C`, `op(X) = Xᵀ` iff the matching
//! flag is `true`, matching `numrs2::blas::gemm`/`gemv` (the pure-Rust
//! reference implementation) and standard BLAS GEMM/GEMV convention: when
//! `trans_a`/`trans_b`/`trans` is set, the operand array is expected to
//! hold the *physical* (already-transposed) storage, and `op(X)` un-does
//! that storage transpose.
//!
//! # Expected-value provenance
//!
//! All expected products below were computed by hand and cross-checked
//! with NumPy:
//!
//! ```python
//! import numpy as np
//! A = np.array([[1.,2.,3.],[4.,5.,6.]])          # 2x3
//! B = np.array([[7.,8.],[9.,10.],[11.,12.]])     # 3x2
//! A @ B  # -> [[58., 64.], [139., 154.]]
//!
//! x3 = np.array([2.,1.,3.])
//! A @ x3  # -> [13., 31.]
//!
//! x2 = np.array([2.,3.])
//! A.T @ x2  # -> [14., 19., 24.]
//! ```

use approx::assert_relative_eq;
use numrs2::blas;
use numrs2::linalg_accelerated::AcceleratedBlas;
use numrs2::prelude::*;

const EPS: f64 = 1e-10;

// ============================================================================
// Shared fixtures (non-symmetric, all-distinct-entries 2x3 / 3x2 matrices)
// ============================================================================

/// Logical A operand, shape (2, 3). Used directly when `trans_a = false`.
fn a_log() -> Array<f64> {
    Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3])
}

/// Physical storage of A for `trans_a = true`: shape (3, 2), the transpose
/// of `a_log()`. `op(A) = a_stored_t().t() == a_log()`.
fn a_stored_t() -> Array<f64> {
    Array::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).reshape(&[3, 2])
}

/// Logical B operand, shape (3, 2). Used directly when `trans_b = false`.
fn b_log() -> Array<f64> {
    Array::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&[3, 2])
}

/// Physical storage of B for `trans_b = true`: shape (2, 3), the transpose
/// of `b_log()`. `op(B) = b_stored_t().t() == b_log()`.
fn b_stored_t() -> Array<f64> {
    Array::from_vec(vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0]).reshape(&[2, 3])
}

/// `a_log() @ b_log()`, hand-computed and NumPy-verified (see module docs).
const AB_EXPECTED: [[f64; 2]; 2] = [[58.0, 64.0], [139.0, 154.0]];

fn assert_2x2_eq(actual: &Array<f64>, expected: &[[f64; 2]; 2], msg: &str) {
    assert_eq!(actual.shape(), vec![2, 2], "{msg}: unexpected shape");
    for i in 0..2 {
        for j in 0..2 {
            let got = actual.get(&[i, j]).expect("in-bounds index");
            assert_relative_eq!(got, expected[i][j], epsilon = EPS);
        }
    }
}

// ============================================================================
// gemm: all 4 trans_a/trans_b combinations
// ============================================================================

#[test]
fn test_gemm_no_transpose_matches_hand_computed_product() {
    let a = a_log();
    let b = b_log();
    let mut c = Array::<f64>::zeros(&[2, 2]);

    AcceleratedBlas::gemm(&a, &b, &mut c, 1.0, 0.0, false, false)
        .expect("gemm(false, false) should succeed");

    assert_2x2_eq(&c, &AB_EXPECTED, "trans_a=false, trans_b=false");
}

#[test]
fn test_gemm_transpose_a_only_matches_hand_computed_product() {
    // A is stored as its (3x2) transpose; trans_a=true must undo that
    // storage transpose so the *mathematical* product is still A_log @ B_log.
    let a = a_stored_t(); // physical shape (3, 2)
    let b = b_log(); // physical shape (3, 2)
    let mut c = Array::<f64>::zeros(&[2, 2]);

    AcceleratedBlas::gemm(&a, &b, &mut c, 1.0, 0.0, true, false)
        .expect("gemm(true, false) should succeed");

    assert_2x2_eq(&c, &AB_EXPECTED, "trans_a=true, trans_b=false");
}

#[test]
fn test_gemm_transpose_b_only_matches_hand_computed_product() {
    let a = a_log(); // physical shape (2, 3)
    let b = b_stored_t(); // physical shape (2, 3)
    let mut c = Array::<f64>::zeros(&[2, 2]);

    AcceleratedBlas::gemm(&a, &b, &mut c, 1.0, 0.0, false, true)
        .expect("gemm(false, true) should succeed");

    assert_2x2_eq(&c, &AB_EXPECTED, "trans_a=false, trans_b=true");
}

#[test]
fn test_gemm_transpose_both_matches_hand_computed_product() {
    // Both operands transposed: physical shapes (3,2) and (2,3) — swapped
    // from the no-transpose case but still conformable to *a* 3x3 product
    // if the flags were ignored (the exact silent-bug scenario), so this
    // combination specifically catches the historical bug returning a
    // wrong-shaped/wrong-valued result instead of erroring outright.
    let a = a_stored_t(); // physical shape (3, 2)
    let b = b_stored_t(); // physical shape (2, 3)
    let mut c = Array::<f64>::zeros(&[2, 2]);

    AcceleratedBlas::gemm(&a, &b, &mut c, 1.0, 0.0, true, true)
        .expect("gemm(true, true) should succeed");

    assert_2x2_eq(&c, &AB_EXPECTED, "trans_a=true, trans_b=true");
}

#[test]
fn test_gemm_transpose_a_with_beta_and_prefilled_c() {
    // alpha=2, beta=3, C pre-filled with 1s:
    // expected = 2 * (A_log @ B_log) + 3 * [[1,1],[1,1]]
    //          = [[119, 131], [281, 311]]  (NumPy-verified)
    let a = a_stored_t();
    let b = b_log();
    let mut c = Array::from_vec(vec![1.0, 1.0, 1.0, 1.0]).reshape(&[2, 2]);

    AcceleratedBlas::gemm(&a, &b, &mut c, 2.0, 3.0, true, false)
        .expect("gemm(true, false) with beta should succeed");

    assert_2x2_eq(
        &c,
        &[[119.0, 131.0], [281.0, 311.0]],
        "trans_a=true, trans_b=false, alpha=2, beta=3",
    );
}

#[test]
fn test_gemm_transpose_all_combos_match_naive_blas_reference() {
    // Cross-check AcceleratedBlas::gemm against numrs2::blas::gemm (the
    // pure-Rust reference implementation, which already honors trans_a/
    // trans_b correctly) for every flag combination.
    let combos: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];

    for (trans_a, trans_b) in combos {
        let a = if trans_a { a_stored_t() } else { a_log() };
        let b = if trans_b { b_stored_t() } else { b_log() };

        let mut c_acc = Array::<f64>::zeros(&[2, 2]);
        AcceleratedBlas::gemm(&a, &b, &mut c_acc, 1.0, 0.0, trans_a, trans_b)
            .unwrap_or_else(|e| panic!("AcceleratedBlas::gemm({trans_a}, {trans_b}) failed: {e}"));

        let mut c_ref = Array::<f64>::zeros(&[2, 2]);
        blas::gemm(&a, &b, &mut c_ref, 1.0, 0.0, trans_a, trans_b)
            .unwrap_or_else(|e| panic!("blas::gemm({trans_a}, {trans_b}) failed: {e}"));

        for i in 0..2 {
            for j in 0..2 {
                let acc_val = c_acc.get(&[i, j]).expect("in-bounds index");
                let ref_val = c_ref.get(&[i, j]).expect("in-bounds index");
                assert_relative_eq!(acc_val, ref_val, epsilon = EPS);
                assert_relative_eq!(acc_val, AB_EXPECTED[i][j], epsilon = EPS);
            }
        }
    }
}

// ============================================================================
// gemv: both trans combinations, including a 3x2-physical-matrix variant
// ============================================================================

#[test]
fn test_gemv_no_transpose_matches_hand_computed_product() {
    // A (2x3) @ x(len 3) -> y(len 2) = [13, 31] (NumPy-verified).
    let a = a_log();
    let x = Array::from_vec(vec![2.0, 1.0, 3.0]);
    let mut y = Array::<f64>::zeros(&[2]);

    AcceleratedBlas::gemv(&a, &x, &mut y, 1.0, 0.0, false).expect("gemv(false) should succeed");

    assert_relative_eq!(y.get(&[0]).expect("index"), 13.0, epsilon = EPS);
    assert_relative_eq!(y.get(&[1]).expect("index"), 31.0, epsilon = EPS);
}

#[test]
fn test_gemv_transpose_matches_hand_computed_product() {
    // A physically (2x3); trans=true means op(A) = A^T (3x2).
    // A^T @ x(len 2) -> y(len 3) = [14, 19, 24] (NumPy-verified).
    let a = a_log();
    let x = Array::from_vec(vec![2.0, 3.0]);
    let mut y = Array::<f64>::zeros(&[3]);

    AcceleratedBlas::gemv(&a, &x, &mut y, 1.0, 0.0, true).expect("gemv(true) should succeed");

    assert_relative_eq!(y.get(&[0]).expect("index"), 14.0, epsilon = EPS);
    assert_relative_eq!(y.get(&[1]).expect("index"), 19.0, epsilon = EPS);
    assert_relative_eq!(y.get(&[2]).expect("index"), 24.0, epsilon = EPS);
}

#[test]
fn test_gemv_no_transpose_with_beta_and_prefilled_y() {
    // alpha=2, beta=-1, y pre-filled [10, 20]:
    // expected = 2 * [13, 31] - [10, 20] = [16, 42] (NumPy-verified).
    let a = a_log();
    let x = Array::from_vec(vec![2.0, 1.0, 3.0]);
    let mut y = Array::from_vec(vec![10.0, 20.0]);

    AcceleratedBlas::gemv(&a, &x, &mut y, 2.0, -1.0, false)
        .expect("gemv(false) with beta should succeed");

    assert_relative_eq!(y.get(&[0]).expect("index"), 16.0, epsilon = EPS);
    assert_relative_eq!(y.get(&[1]).expect("index"), 42.0, epsilon = EPS);
}

#[test]
fn test_gemv_transpose_with_beta_and_prefilled_y() {
    // alpha=0.5, beta=2, y pre-filled [1, 2, 3]:
    // expected = 0.5 * [14, 19, 24] + 2 * [1, 2, 3] = [9, 13.5, 18]
    // (NumPy-verified).
    let a = a_log();
    let x = Array::from_vec(vec![2.0, 3.0]);
    let mut y = Array::from_vec(vec![1.0, 2.0, 3.0]);

    AcceleratedBlas::gemv(&a, &x, &mut y, 0.5, 2.0, true)
        .expect("gemv(true) with beta should succeed");

    assert_relative_eq!(y.get(&[0]).expect("index"), 9.0, epsilon = EPS);
    assert_relative_eq!(y.get(&[1]).expect("index"), 13.5, epsilon = EPS);
    assert_relative_eq!(y.get(&[2]).expect("index"), 18.0, epsilon = EPS);
}

#[test]
fn test_gemv_3x2_physical_matrix_both_transpose_flags() {
    // Physical A this time has more rows than columns (3x2), covering the
    // opposite aspect ratio from `a_log()`.
    // A = [[1,2],[3,4],[5,6]] (3x2).
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[3, 2]);

    // trans=false: A(3x2) @ x(len 2) -> y(len 3) = [5, 11, 17].
    let x_false = Array::from_vec(vec![1.0, 2.0]);
    let mut y_false = Array::<f64>::zeros(&[3]);
    AcceleratedBlas::gemv(&a, &x_false, &mut y_false, 1.0, 0.0, false)
        .expect("gemv(false) on 3x2 matrix should succeed");
    assert_relative_eq!(y_false.get(&[0]).expect("index"), 5.0, epsilon = EPS);
    assert_relative_eq!(y_false.get(&[1]).expect("index"), 11.0, epsilon = EPS);
    assert_relative_eq!(y_false.get(&[2]).expect("index"), 17.0, epsilon = EPS);

    // trans=true: A^T(2x3) @ x(len 3) -> y(len 2) = [22, 28].
    let x_true = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let mut y_true = Array::<f64>::zeros(&[2]);
    AcceleratedBlas::gemv(&a, &x_true, &mut y_true, 1.0, 0.0, true)
        .expect("gemv(true) on 3x2 matrix should succeed");
    assert_relative_eq!(y_true.get(&[0]).expect("index"), 22.0, epsilon = EPS);
    assert_relative_eq!(y_true.get(&[1]).expect("index"), 28.0, epsilon = EPS);
}

#[test]
fn test_gemv_both_transpose_flags_match_naive_blas_reference() {
    // Cross-check AcceleratedBlas::gemv against numrs2::blas::gemv (the
    // pure-Rust reference implementation) for both flag values.
    let a = a_log();

    for trans in [false, true] {
        let (x, y_len) = if trans {
            (Array::from_vec(vec![2.0, 3.0]), 3)
        } else {
            (Array::from_vec(vec![2.0, 1.0, 3.0]), 2)
        };

        let mut y_acc = Array::<f64>::zeros(&[y_len]);
        AcceleratedBlas::gemv(&a, &x, &mut y_acc, 1.0, 0.0, trans)
            .unwrap_or_else(|e| panic!("AcceleratedBlas::gemv(trans={trans}) failed: {e}"));

        let mut y_ref = Array::<f64>::zeros(&[y_len]);
        blas::gemv(&a, &x, &mut y_ref, 1.0, 0.0, trans)
            .unwrap_or_else(|e| panic!("blas::gemv(trans={trans}) failed: {e}"));

        for i in 0..y_len {
            let acc_val = y_acc.get(&[i]).expect("in-bounds index");
            let ref_val = y_ref.get(&[i]).expect("in-bounds index");
            assert_relative_eq!(acc_val, ref_val, epsilon = EPS);
        }
    }
}
