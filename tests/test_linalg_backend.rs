//! Correctness harness for the `scirs2-linalg` fast-path backend behind
//! `Array::{det, inv, solve, svd, eig, cholesky, qr}`
//! (`src/linalg/backend.rs`).
//!
//! # What these tests check, and what they deliberately do not
//!
//! The backend engages only for `f32`/`f64` operands at `n >= 16`
//! (`n >= 4` for `solve`). Below that, the pre-existing closed-form / LU
//! code runs. Both paths must satisfy the *same mathematical contract*,
//! so every assertion here is an **invariant** -- `A A^-1 = I`, `Q R = A`
//! with `Q^T Q = I`, `U S V^T = A`, `L L^T = A`, `A v = lambda v` -- and
//! never an elementwise comparison against a stored expected matrix.
//!
//! That is not laziness: `Q`/`U`/`V`/eigenvector columns are each only
//! determined up to sign (and, for repeated singular/eigenvalues, up to
//! rotation within a subspace). LAPACK and a hand-rolled Householder
//! sweep routinely disagree on those signs while both being correct, so
//! a sign-sensitive assertion would fail for reasons that have nothing to
//! do with correctness.
//!
//! Sizes are chosen to straddle the gate: `n = 4` exercises the old path,
//! `n = 16` sits exactly on the threshold, and `n = 64` is well past it.

#![cfg(all(feature = "matrix_decomp", feature = "lapack"))]

use numrs2::prelude::*;
use scirs2_core::random::Random;

/// Invariants hold to near machine precision on the `f64` path.
const TOL_F64: f64 = 1e-10;
/// `f32` carries ~7 decimal digits; 1e-4 is the usual working tolerance.
const TOL_F32: f32 = 1e-4;

// ---------------------------------------------------------------------
// Fixtures (deterministic: fixed seeds via scirs2_core::random)
// ---------------------------------------------------------------------

/// Deterministic, strictly diagonally dominant -- hence non-singular and
/// well-conditioned -- `n x n` matrix. Non-symmetric by construction.
fn well_conditioned(n: usize, seed: u64) -> Array<f64> {
    let mut rng = Random::seed(seed);
    let mut data = vec![0.0_f64; n * n];
    for (idx, slot) in data.iter_mut().enumerate() {
        let (i, j) = (idx / n, idx % n);
        let v: f64 = rng.gen_range(-1.0..1.0);
        *slot = if i == j { v + n as f64 } else { v };
    }
    Array::from_vec(data).reshape(&[n, n])
}

/// Deterministic symmetric positive-definite `n x n` matrix, built as
/// `A^T A + n I`.
///
/// `A^T A` is *exactly* symmetric here (not merely symmetric to within
/// rounding): entry `(i, j)` and entry `(j, i)` are the same sum of the
/// same products accumulated in the same order, so they agree
/// bit-for-bit. That matters because the backend's `eig` gate requires
/// exact symmetry.
///
/// # Do not raise the sizes this is called with past ~200
///
/// This fixture is built with `matmul`, which is currently *silently
/// wrong* at `n = 256` (verified: correct at `n <= 200`, max elementwise
/// error 2.3e3 at 256, with both contiguous and transposed left
/// operands). Every size used here is well inside the correct range, and
/// `sym_pd_direct` below exists for the two tests that need a large
/// symmetric matrix.
fn spd(n: usize, seed: u64) -> Array<f64> {
    let a = well_conditioned(n, seed);
    let mut m = a.transpose().matmul(&a).expect("matmul should succeed");
    for i in 0..n {
        let v = m.get(&[i, i]).expect("in-bounds get");
        m.set(&[i, i], v + n as f64).expect("in-bounds set");
    }
    m
}

fn rhs(n: usize, seed: u64) -> Array<f64> {
    let mut rng = Random::seed(seed);
    Array::from_vec((0..n).map(|_| rng.gen_range(-1.0..1.0)).collect())
}

fn to_f32(a: &Array<f64>) -> Array<f32> {
    let shape = a.shape();
    let data: Vec<f32> = a.to_vec().into_iter().map(|v| v as f32).collect();
    Array::from_vec(data).reshape(&shape)
}

/// Deterministic, *exactly* symmetric, strictly diagonally dominant
/// (hence positive definite) matrix, built entry-by-entry so that
/// `m[i][j]` and `m[j][i]` evaluate the same expression.
///
/// Unlike [`spd`] this uses no `matmul`, so it stays valid at the large
/// sizes the above-the-cap tests need.
fn sym_pd_direct(n: usize) -> Array<f64> {
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
            let v = (((lo * 31 + hi * 17) % 13) as f64) / 13.0 - 0.5;
            data[i * n + j] = if i == j { v + 2.0 * n as f64 } else { v };
        }
    }
    Array::from_vec(data).reshape(&[n, n])
}

/// The sizes under test: 4 is below the backend gate (old path), 16 is
/// exactly on it, 64 is well past it.
const SIZES: [usize; 3] = [4, 16, 64];

/// Sizes at or above the `qr` / `svd` upper bounds (`QR_MAX_DIM = 96`,
/// `SVD_MAX_DIM = 256`), where those two operations hand back to the
/// existing path. Kept small in count -- an `n = 256` SVD is not cheap --
/// but present, because without them the above-the-cap branch is shipped
/// code with no coverage at all.
const QR_ABOVE_CAP: usize = 128;
const SVD_ABOVE_CAP: usize = 256;

// ---------------------------------------------------------------------
// Independent oracles
// ---------------------------------------------------------------------

/// Determinant by Doolittle LU with partial pivoting, written here from
/// scratch so it shares no code with either path under test.
fn det_lu_oracle(a: &Array<f64>) -> f64 {
    let n = a.shape()[0];
    let mut m = a.to_vec();
    let mut det = 1.0_f64;

    for col in 0..n {
        // Partial pivot: largest magnitude in this column at or below the
        // diagonal.
        let mut pivot = col;
        for row in (col + 1)..n {
            if m[row * n + col].abs() > m[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if m[pivot * n + col] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            for k in 0..n {
                m.swap(col * n + k, pivot * n + k);
            }
            det = -det;
        }
        det *= m[col * n + col];
        for row in (col + 1)..n {
            let factor = m[row * n + col] / m[col * n + col];
            for k in col..n {
                m[row * n + k] -= factor * m[col * n + k];
            }
        }
    }
    det
}

/// Textbook `n x n` matrix product, written here from scratch.
///
/// The above-the-cap tests need this rather than [`Array::matmul`]:
/// `matmul` is currently silently wrong at `n = 256` (see the note on
/// [`spd`]), which is exactly the size `SVD_MAX_DIM` puts under test. A
/// reconstruction check must not be built on the thing that is broken.
fn naive_matmul(a: &Array<f64>, b: &Array<f64>, n: usize) -> Array<f64> {
    let (av, bv) = (a.to_vec(), b.to_vec());
    let mut c = vec![0.0_f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = av[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * bv[k * n + j];
            }
        }
    }
    Array::from_vec(c).reshape(&[n, n])
}

/// Largest absolute difference between `a` and the identity.
fn max_dev_from_identity(a: &Array<f64>) -> f64 {
    let n = a.shape()[0];
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = a.get(&[i, j]).expect("in-bounds get");
            worst = worst.max((actual - expected).abs());
        }
    }
    worst
}

/// Largest absolute elementwise difference between two same-shaped 2-D
/// arrays.
fn max_abs_diff(a: &Array<f64>, b: &Array<f64>) -> f64 {
    a.to_vec()
        .into_iter()
        .zip(b.to_vec())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

fn max_abs_diff_f32(a: &Array<f32>, b: &Array<f32>) -> f32 {
    a.to_vec()
        .into_iter()
        .zip(b.to_vec())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Build the `k x k` diagonal matrix of singular values.
fn diag(values: &Array<f64>, n: usize) -> Array<f64> {
    let mut d = Array::<f64>::zeros(&[n, n]);
    for i in 0..values.shape()[0].min(n) {
        let v = values.get(&[i]).expect("in-bounds get");
        d.set(&[i, i], v).expect("in-bounds set");
    }
    d
}

// ---------------------------------------------------------------------
// det
// ---------------------------------------------------------------------

#[test]
fn det_matches_independent_lu_oracle() {
    for &n in &SIZES {
        let a = well_conditioned(n, 1234 + n as u64);
        let got = a.det().expect("det should succeed");
        let want = det_lu_oracle(&a);

        // Determinants of a diagonally dominant n x n grow like n^n, so
        // the comparison must be relative, not absolute.
        let rel = (got - want).abs() / want.abs().max(1.0);
        assert!(
            rel < 1e-9,
            "n={n}: det = {got:e}, LU oracle = {want:e} (relative error {rel:e})"
        );
    }
}

#[test]
fn det_of_identity_is_one() {
    for &n in &SIZES {
        let mut a = Array::<f64>::zeros(&[n, n]);
        for i in 0..n {
            a.set(&[i, i], 1.0).expect("in-bounds set");
        }
        let d = a.det().expect("det should succeed");
        assert!((d - 1.0).abs() < TOL_F64, "n={n}: det(I) = {d}");
    }
}

#[test]
fn det_rejects_non_square() {
    let a = Array::from_vec(vec![1.0_f64; 16 * 20]).reshape(&[16, 20]);
    let err = a.det().expect_err("non-square det must error");
    assert!(
        err.to_string().contains("square"),
        "unexpected message: {err}"
    );
}

// ---------------------------------------------------------------------
// inv
// ---------------------------------------------------------------------

#[test]
fn inv_round_trips_to_identity() {
    for &n in &SIZES {
        let a = well_conditioned(n, 555 + n as u64);
        let ainv = a.inv().expect("inv should succeed");

        let left = a.matmul(&ainv).expect("matmul should succeed");
        let right = ainv.matmul(&a).expect("matmul should succeed");

        assert!(
            max_dev_from_identity(&left) < TOL_F64,
            "n={n}: max |A A^-1 - I| = {}",
            max_dev_from_identity(&left)
        );
        assert!(
            max_dev_from_identity(&right) < TOL_F64,
            "n={n}: max |A^-1 A - I| = {}",
            max_dev_from_identity(&right)
        );
    }
}

#[test]
fn inv_rejects_non_square() {
    let a = Array::from_vec(vec![1.0_f64; 16 * 20]).reshape(&[16, 20]);
    let err = a.inv().expect_err("non-square inv must error");
    assert!(
        err.to_string().contains("square"),
        "unexpected message: {err}"
    );
}

// ---------------------------------------------------------------------
// solve
// ---------------------------------------------------------------------

/// Regression test for an unconditional infinite recursion.
///
/// Before the backend hook, `Array::solve` for `n > 3` delegated to
/// `interop::scirs_compat::solve_linear_system` -> `linalg::solve` ->
/// `Array::solve`, with no base case. On the unhooked build this test
/// aborted the process (`SIGABRT`, `fatal runtime error: stack
/// overflow`). Reaching the assertions at all is the real assertion.
#[test]
fn solve_does_not_recurse_for_n_above_three() {
    for &n in &[4_usize, 5, 16, 64] {
        let a = well_conditioned(n, 77 + n as u64);
        let x_expected = rhs(n, 88 + n as u64);

        // Derive b = A x so the exact solution is known.
        let x_col = Array::from_vec(x_expected.to_vec()).reshape(&[n, 1]);
        let b_col = a.matmul(&x_col).expect("matmul should succeed");
        let b = Array::from_vec(b_col.to_vec());

        let x = a.solve(&b).expect("solve should succeed");
        assert_eq!(x.shape(), vec![n], "n={n}: solution shape");

        let worst = x
            .to_vec()
            .into_iter()
            .zip(x_expected.to_vec())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst < TOL_F64, "n={n}: max |x - x_exact| = {worst}");
    }
}

#[test]
fn solve_residual_is_small() {
    for &n in &SIZES {
        let a = well_conditioned(n, 909 + n as u64);
        let b = rhs(n, 313 + n as u64);
        let x = a.solve(&b).expect("solve should succeed");

        let x_col = Array::from_vec(x.to_vec()).reshape(&[n, 1]);
        let ax = a.matmul(&x_col).expect("matmul should succeed");

        let worst = ax
            .to_vec()
            .into_iter()
            .zip(b.to_vec())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst < TOL_F64, "n={n}: max |A x - b| = {worst}");
    }
}

#[test]
fn solve_rejects_mismatched_rhs() {
    let a = well_conditioned(16, 1);
    let b = rhs(15, 2);
    assert!(
        a.solve(&b).is_err(),
        "a 15-element rhs against a 16x16 matrix must error"
    );
}

// ---------------------------------------------------------------------
// svd
// ---------------------------------------------------------------------

#[test]
fn svd_reconstructs_and_has_sorted_nonnegative_values() {
    for &n in &SIZES {
        let a = well_conditioned(n, 4242 + n as u64);
        let (u, s, vt) = a.svd().expect("svd should succeed");

        assert_eq!(u.shape(), vec![n, n], "n={n}: U shape");
        assert_eq!(vt.shape(), vec![n, n], "n={n}: V^T shape");
        assert_eq!(s.shape(), vec![n], "n={n}: singular value count");

        // Reconstruction: U diag(s) V^T == A.
        let recon = u
            .matmul(&diag(&s, n))
            .expect("matmul should succeed")
            .matmul(&vt)
            .expect("matmul should succeed");
        let worst = max_abs_diff(&recon, &a);
        assert!(
            worst < 1e-9 * (n as f64),
            "n={n}: max |U S V^T - A| = {worst}"
        );

        // Singular values are non-negative and descending.
        let sv = s.to_vec();
        for (i, &v) in sv.iter().enumerate() {
            assert!(v >= -TOL_F64, "n={n}: singular value [{i}] = {v} < 0");
        }
        for w in sv.windows(2) {
            assert!(
                w[0] >= w[1] - TOL_F64,
                "n={n}: singular values not descending: {:?}",
                sv
            );
        }

        // U and V^T are orthogonal.
        assert!(
            max_dev_from_identity(&u.transpose().matmul(&u).expect("matmul")) < 1e-9,
            "n={n}: U is not orthogonal"
        );
        assert!(
            max_dev_from_identity(&vt.matmul(&vt.transpose()).expect("matmul")) < 1e-9,
            "n={n}: V^T is not orthogonal"
        );
    }
}

// ---------------------------------------------------------------------
// qr
// ---------------------------------------------------------------------

#[test]
fn qr_reconstructs_with_orthogonal_q_and_upper_triangular_r() {
    for &n in &SIZES {
        let a = well_conditioned(n, 31337 + n as u64);
        let (q, r) = a.qr().expect("qr should succeed");

        assert_eq!(q.shape(), vec![n, n], "n={n}: Q shape");
        assert_eq!(r.shape(), vec![n, n], "n={n}: R shape");

        // Q R == A.
        let recon = q.matmul(&r).expect("matmul should succeed");
        let worst = max_abs_diff(&recon, &a);
        assert!(worst < 1e-9 * (n as f64), "n={n}: max |Q R - A| = {worst}");

        // Q^T Q == I (orthogonality; sign-agnostic by construction).
        let qtq = q.transpose().matmul(&q).expect("matmul should succeed");
        assert!(
            max_dev_from_identity(&qtq) < 1e-9,
            "n={n}: max |Q^T Q - I| = {}",
            max_dev_from_identity(&qtq)
        );

        // R is upper triangular.
        for i in 1..n {
            for j in 0..i {
                let v = r.get(&[i, j]).expect("in-bounds get");
                assert!(v.abs() < 1e-9, "n={n}: R[{i},{j}] = {v} should be 0");
            }
        }
    }
}

/// Above `QR_MAX_DIM` the backend declines and the existing Householder
/// path runs. That branch is only reachable because of the cap, so it
/// gets the same invariants -- and this test is what catches a typo'd
/// bound.
#[test]
fn qr_above_the_cap_still_satisfies_its_invariants() {
    let n = QR_ABOVE_CAP;
    let a = sym_pd_direct(n);
    let (q, r) = a.qr().expect("qr above the cap should succeed");

    assert_eq!(q.shape(), vec![n, n], "Q shape");
    assert_eq!(r.shape(), vec![n, n], "R shape");

    let recon = naive_matmul(&q, &r, n);
    let scale = a.to_vec().into_iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = max_abs_diff(&recon, &a);
    assert!(
        worst < 1e-9 * scale.max(1.0) * (n as f64),
        "n={n}: max |Q R - A| = {worst}"
    );

    let qtq = naive_matmul(&q.transpose(), &q, n);
    assert!(
        max_dev_from_identity(&qtq) < 1e-9,
        "n={n}: max |Q^T Q - I| = {}",
        max_dev_from_identity(&qtq)
    );

    for i in 1..n {
        for j in 0..i {
            let v = r.get(&[i, j]).expect("in-bounds get");
            assert!(v.abs() < 1e-8, "n={n}: R[{i},{j}] = {v} should be 0");
        }
    }
}

/// Same idea for `svd` above `SVD_MAX_DIM`.
#[test]
fn svd_above_the_cap_still_satisfies_its_invariants() {
    let n = SVD_ABOVE_CAP;
    let a = sym_pd_direct(n);
    let (u, s, vt) = a.svd().expect("svd above the cap should succeed");

    assert_eq!(u.shape(), vec![n, n], "U shape");
    assert_eq!(vt.shape(), vec![n, n], "V^T shape");
    assert_eq!(s.shape(), vec![n], "singular value count");

    let us = naive_matmul(&u, &diag(&s, n), n);
    let recon = naive_matmul(&us, &vt, n);
    let scale = a.to_vec().into_iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = max_abs_diff(&recon, &a);
    assert!(
        worst < 1e-9 * scale.max(1.0) * (n as f64),
        "n={n}: max |U S V^T - A| = {worst}"
    );

    for (i, &v) in s.to_vec().iter().enumerate() {
        assert!(v >= -TOL_F64, "n={n}: singular value [{i}] = {v} < 0");
    }
}

// ---------------------------------------------------------------------
// cholesky
// ---------------------------------------------------------------------

#[test]
fn cholesky_factor_is_lower_triangular_and_reconstructs() {
    for &n in &SIZES {
        let a = spd(n, 2024 + n as u64);
        let l = a.cholesky().expect("cholesky should succeed");

        assert_eq!(l.shape(), vec![n, n], "n={n}: L shape");

        // L is lower triangular.
        for i in 0..n {
            for j in (i + 1)..n {
                let v = l.get(&[i, j]).expect("in-bounds get");
                assert!(v.abs() < 1e-9, "n={n}: L[{i},{j}] = {v} should be 0");
            }
        }

        // L L^T == A.
        let recon = l.matmul(&l.transpose()).expect("matmul should succeed");
        let worst = max_abs_diff(&recon, &a);
        // A's entries scale like n^2 here, so scale the tolerance with it.
        let scale = a.to_vec().into_iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            worst < 1e-10 * scale.max(1.0) * (n as f64),
            "n={n}: max |L L^T - A| = {worst} (scale {scale})"
        );
    }
}

// ---------------------------------------------------------------------
// eig
// ---------------------------------------------------------------------

#[test]
fn eig_pairs_satisfy_the_eigen_equation() {
    for &n in &SIZES {
        let a = spd(n, 606 + n as u64);
        let (vals, vecs) = a.eig().expect("eig should succeed");

        assert_eq!(vals.shape(), vec![n], "n={n}: eigenvalue count");
        assert_eq!(vecs.shape(), vec![n, n], "n={n}: eigenvector matrix shape");

        // Column k of `vecs` is the eigenvector for eigenvalue k:
        // check A v = lambda v directly, which is sign- and
        // order-agnostic.
        let scale = a.to_vec().into_iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for k in 0..n {
            let lambda = vals.get(&[k]).expect("in-bounds get");
            let mut v = Array::<f64>::zeros(&[n, 1]);
            for i in 0..n {
                v.set(&[i, 0], vecs.get(&[i, k]).expect("in-bounds get"))
                    .expect("in-bounds set");
            }
            let av = a.matmul(&v).expect("matmul should succeed");

            let worst = av
                .to_vec()
                .into_iter()
                .zip(v.to_vec())
                .map(|(lhs, rhs)| (lhs - lambda * rhs).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                worst < 1e-8 * scale.max(1.0),
                "n={n}, pair {k}: max |A v - lambda v| = {worst} (lambda = {lambda})"
            );
        }

        // All eigenvalues of an SPD matrix are positive.
        for (i, lambda) in vals.to_vec().into_iter().enumerate() {
            assert!(lambda > 0.0, "n={n}: SPD eigenvalue [{i}] = {lambda} <= 0");
        }
    }
}

/// Path witness: the *only* externally visible difference between the
/// backend and the old path for symmetric `eig` is eigenvalue ordering.
///
/// The pre-existing Wilkinson-shifted QR iteration returns them in an
/// order with no rule to it (measured on this exact fixture at `n = 16`:
/// `179.7, 353.6, 195.5, 343.7, ...` -- neither ascending, descending,
/// nor by magnitude). The backend normalises to descending algebraic
/// order, matching the descending singular values `svd` already returns.
///
/// So a monotone result at `n >= 16` is proof the backend actually ran,
/// and this test would fail if the hook silently stopped engaging.
#[test]
fn eig_is_descending_on_the_backend_path() {
    for &n in &[16_usize, 64] {
        let a = spd(n, 606 + n as u64);
        let (vals, _) = a.eig().expect("eig should succeed");
        let v = vals.to_vec();
        assert!(
            v.windows(2).all(|w| w[0] >= w[1]),
            "n={n}: backend eig should return descending eigenvalues, got {v:?}"
        );
    }
}

/// For a symmetric positive-definite matrix the eigenvalues and the
/// singular values are the same numbers. This pins the ordering
/// consistency that motivates the backend's descending choice: the two
/// methods must not disagree on direction.
#[test]
fn eig_and_svd_agree_on_spd_input() {
    for &n in &[16_usize, 64] {
        let a = spd(n, 606 + n as u64);
        let (vals, _) = a.eig().expect("eig should succeed");
        let (_, sv, _) = a.svd().expect("svd should succeed");

        let scale = a.to_vec().into_iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (i, (lambda, sigma)) in vals.to_vec().into_iter().zip(sv.to_vec()).enumerate() {
            assert!(
                (lambda - sigma).abs() < 1e-8 * scale.max(1.0),
                "n={n}: eigenvalue[{i}] = {lambda} but singular value[{i}] = {sigma}"
            );
        }
    }
}

#[test]
fn eig_of_non_symmetric_input_still_works() {
    // Non-symmetric input is declined by the backend and must fall
    // through to the existing QR iteration without error.
    let a = well_conditioned(16, 5150);
    let (vals, vecs) = a.eig().expect("non-symmetric eig should still succeed");
    assert_eq!(vals.shape(), vec![16]);
    assert_eq!(vecs.shape(), vec![16, 16]);
}

// ---------------------------------------------------------------------
// f32 coverage
// ---------------------------------------------------------------------

#[test]
fn f32_inv_round_trips_to_identity() {
    for &n in &SIZES {
        let a = to_f32(&well_conditioned(n, 21 + n as u64));
        let ainv = a.inv().expect("f32 inv should succeed");
        let prod = a.matmul(&ainv).expect("matmul should succeed");

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0_f32 } else { 0.0 };
                let actual = prod.get(&[i, j]).expect("in-bounds get");
                assert!(
                    (actual - expected).abs() < TOL_F32,
                    "f32 n={n}: (A A^-1)[{i},{j}] = {actual}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn f32_qr_reconstructs_with_orthogonal_q() {
    for &n in &SIZES {
        let a = to_f32(&well_conditioned(n, 3131 + n as u64));
        let (q, r) = a.qr().expect("f32 qr should succeed");

        let recon = q.matmul(&r).expect("matmul should succeed");
        let worst = max_abs_diff_f32(&recon, &a);
        assert!(
            worst < TOL_F32 * (n as f32),
            "f32 n={n}: max |Q R - A| = {worst}"
        );

        let qtq = q.transpose().matmul(&q).expect("matmul should succeed");
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0_f32 } else { 0.0 };
                let actual = qtq.get(&[i, j]).expect("in-bounds get");
                assert!(
                    (actual - expected).abs() < TOL_F32,
                    "f32 n={n}: (Q^T Q)[{i},{j}] = {actual}"
                );
            }
        }
    }
}

#[test]
fn f32_svd_reconstructs() {
    for &n in &SIZES {
        let a = to_f32(&well_conditioned(n, 8080 + n as u64));
        let (u, s, vt) = a.svd().expect("f32 svd should succeed");

        let mut d = Array::<f32>::zeros(&[n, n]);
        for i in 0..s.shape()[0].min(n) {
            d.set(&[i, i], s.get(&[i]).expect("in-bounds get"))
                .expect("in-bounds set");
        }
        let recon = u
            .matmul(&d)
            .expect("matmul should succeed")
            .matmul(&vt)
            .expect("matmul should succeed");
        let worst = max_abs_diff_f32(&recon, &a);
        assert!(
            worst < TOL_F32 * (n as f32),
            "f32 n={n}: max |U S V^T - A| = {worst}"
        );
    }
}

#[test]
fn f32_solve_residual_is_small() {
    for &n in &SIZES {
        let a64 = well_conditioned(n, 606 + n as u64);
        let a = to_f32(&a64);
        let b = Array::from_vec(
            rhs(n, 707 + n as u64)
                .to_vec()
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<f32>>(),
        );

        let x = a.solve(&b).expect("f32 solve should succeed");
        let x_col = Array::from_vec(x.to_vec()).reshape(&[n, 1]);
        let ax = a.matmul(&x_col).expect("matmul should succeed");

        let worst = ax
            .to_vec()
            .into_iter()
            .zip(b.to_vec())
            .map(|(got, want)| (got - want).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < TOL_F32, "f32 n={n}: max |A x - b| = {worst}");
    }
}

#[test]
fn f32_det_is_close_to_the_f64_determinant() {
    for &n in &[4_usize, 16] {
        let a64 = well_conditioned(n, 99 + n as u64);
        let a32 = to_f32(&a64);

        let d64 = a64.det().expect("f64 det should succeed");
        let d32 = a32.det().expect("f32 det should succeed");

        let rel = ((d32 as f64) - d64).abs() / d64.abs().max(1.0);
        assert!(
            rel < 1e-3,
            "f32 n={n}: det = {d32:e} vs f64 {d64:e} (relative {rel:e})"
        );
    }
}

// ---------------------------------------------------------------------
// Non-f32/f64 element types
// ---------------------------------------------------------------------

/// Compile-level check that the backend's `TypeId` dispatch (and the
/// `T: 'static` bound it needs) did not restrict what integer arrays can
/// do.
///
/// `Array<i64>` never had `det`/`inv`/`svd`/... in the first place --
/// those live in a `T: Float` impl block -- so there is nothing here for
/// the backend to decline. What this pins down is that an `i64` array
/// still builds and computes through the shared array machinery.
#[test]
fn integer_arrays_still_compile_and_compute() {
    let a = Array::from_vec(vec![1_i64, 2, 3, 4]).reshape(&[2, 2]);
    let b = Array::from_vec(vec![1_i64, 0, 0, 1]).reshape(&[2, 2]);

    let c = a.matmul(&b).expect("integer matmul should succeed");
    assert_eq!(c.to_vec(), vec![1_i64, 2, 3, 4]);
    assert_eq!(a.transpose().to_vec(), vec![1_i64, 3, 2, 4]);
}

// ---------------------------------------------------------------------
// Timing harness (manual; not part of the normal run)
// ---------------------------------------------------------------------

/// Run with:
/// `cargo nextest run --release --test test_linalg_backend --run-ignored all -E 'test(timing)' --no-capture`
#[test]
#[ignore = "timing harness; run manually with --run-ignored all"]
fn timing_harness() {
    use std::time::Instant;

    fn bench<F: FnMut()>(label: &str, reps: usize, mut f: F) {
        f(); // warm-up
        let start = Instant::now();
        for _ in 0..reps {
            f();
        }
        println!("{label}\t{:?}", start.elapsed() / reps as u32);
    }

    for &n in &[64_usize, 128] {
        let a = well_conditioned(n, 12345);
        let s = spd(n, 999);
        let b = rhs(n, 4242);
        let reps = if n == 64 { 20 } else { 5 };

        bench(&format!("det n={n}"), reps, || {
            let _ = a.det().expect("det");
        });
        bench(&format!("inv n={n}"), reps, || {
            let _ = a.inv().expect("inv");
        });
        bench(&format!("svd n={n}"), reps, || {
            let _ = a.svd().expect("svd");
        });
        bench(&format!("qr n={n}"), reps, || {
            let _ = a.qr().expect("qr");
        });
        bench(&format!("cholesky n={n}"), reps, || {
            let _ = s.cholesky().expect("cholesky");
        });
        // `solve` is opt-out because the *pre-hook* path is not slow, it
        // is fatal: at `n >= 4` `Array::solve` recurses into itself
        // through `scirs_compat::solve_linear_system` -> `linalg::solve`
        // and aborts the process with a stack overflow (see
        // `backend::SOLVE_MIN_DIM`). To take "before" numbers for the
        // other five operations, set `NUMRS2_TIMING_NO_SOLVE=1` so this
        // block is skipped rather than killing the run.
        if std::env::var_os("NUMRS2_TIMING_NO_SOLVE").is_none() {
            bench(&format!("solve n={n}"), reps, || {
                let _ = a.solve(&b).expect("solve");
            });
        }
    }
}

/// Separate from [`timing_harness`] because the *old* `eig` path is up to
/// 1000 QR iterations with two full matmuls each, and is unusable well
/// before `n = 128`.
#[test]
#[ignore = "timing harness; run manually with --run-ignored all"]
fn timing_harness_eig() {
    use std::time::Instant;

    for &n in &[16_usize, 32, 64] {
        let s = spd(n, 999);
        let start = Instant::now();
        let _ = s.eig().expect("eig");
        println!("eig(sym) n={n}\t{:?}", start.elapsed());
    }
}
