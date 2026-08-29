//! Regression tests for `solve_lyapunov` (continuous-time Lyapunov equation solver).
//!
//! `solve_lyapunov` is documented (and required by callers doing stability analysis) to
//! solve `A*X + X*A^T = -Q` -- note the transpose on the second `A`. A prior bug in the
//! Kronecker-product assembly used `A*X + X*A = -Q` (no transpose) instead, which only
//! coincidentally produced the right answer when `A` happened to be symmetric (where
//! `A^T == A`). These tests exercise the non-symmetric case directly, keep the
//! already-correct symmetric/diagonal cases working, and check the Lyapunov stability
//! certificate (X symmetric positive definite for stable A).

use numrs2::new_modules::control::solve_lyapunov;
use scirs2_core::ndarray::{array, Array2};

/// Max-abs entry of `A*X + X*A^T + Q`, which should be ~0 for a correct solution of
/// `A*X + X*A^T = -Q`.
fn lyapunov_residual(a: &Array2<f64>, x: &Array2<f64>, q: &Array2<f64>) -> f64 {
    let residual: Array2<f64> = a.dot(x) + x.dot(&a.t()) + q;
    residual.iter().fold(0.0_f64, |max, &v| max.max(v.abs()))
}

/// (a) Non-symmetric A: A*X + X*A^T = -Q must hold. This is exactly the case the original
/// bug got wrong -- `a[[l, j]]` (used by the buggy code) differs from the required
/// `a[[j, l]]` whenever A is not symmetric.
#[test]
fn test_solve_lyapunov_nonsymmetric_a_residual() {
    let a: Array2<f64> = array![[-2.0, 1.0], [0.0, -3.0]];
    let q: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]];

    let x = solve_lyapunov(&a, &q)
        .expect("solve_lyapunov should succeed for a stable, non-symmetric A");

    let residual = lyapunov_residual(&a, &x, &q);
    assert!(
        residual < 1e-8,
        "residual max|A X + X A^T + Q| = {residual} exceeds tolerance; X = {x:?}"
    );
}

/// (b) Symmetric A: A^T == A, so this case was already correct before the fix (both
/// `a[[l, j]]` and `a[[j, l]]` read the same value). The fix must not disturb it.
#[test]
fn test_solve_lyapunov_symmetric_a() {
    let a: Array2<f64> = array![[-2.0, 1.0], [1.0, -3.0]];
    let q: Array2<f64> = array![[2.0, 0.5], [0.5, 1.0]];

    let x =
        solve_lyapunov(&a, &q).expect("solve_lyapunov should succeed for a stable, symmetric A");

    let residual = lyapunov_residual(&a, &x, &q);
    assert!(
        residual < 1e-8,
        "residual max|A X + X A^T + Q| = {residual} exceeds tolerance; X = {x:?}"
    );
}

/// (c) Known analytic case. For diagonal `A = diag(a1, a2)` and diagonal `Q = diag(q1, q2)`,
/// `A*X + X*A^T = -Q` reduces elementwise to `X[i,j]*(a_i + a_j) = -Q[i,j]`, so the solution
/// is diagonal with `X[i,i] = -Q[i,i] / (2*a_i)`.
/// `A = diag(-1, -2)`, `Q = diag(2, 4)` => `X = diag(1, 1) = I`.
#[test]
fn test_solve_lyapunov_diagonal_analytic() {
    let a: Array2<f64> = array![[-1.0, 0.0], [0.0, -2.0]];
    let q: Array2<f64> = array![[2.0, 0.0], [0.0, 4.0]];

    let x = solve_lyapunov(&a, &q).expect("solve_lyapunov should succeed for diagonal A");

    let expected: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]];
    for i in 0..2 {
        for j in 0..2 {
            let got = x[[i, j]];
            let want = expected[[i, j]];
            assert!(
                (got - want).abs() < 1e-8,
                "X[{i},{j}] = {got} does not match analytic expectation {want}"
            );
        }
    }
}

/// (d) Stability integration: for stable A (all eigenvalues have negative real part) and
/// positive definite Q, the Lyapunov solution X must itself be symmetric positive definite
/// -- this is the standard Lyapunov stability certificate. Reuses the non-symmetric, stable
/// A from (a) so this also exercises the transpose fix from a different angle (an
/// asymmetric residual bug can easily produce a non-symmetric or indefinite X).
#[test]
fn test_solve_lyapunov_stable_a_gives_spd_x() {
    let a: Array2<f64> = array![[-2.0, 1.0], [0.0, -3.0]];
    let q: Array2<f64> = array![[1.0, 0.0], [0.0, 1.0]];

    let x = solve_lyapunov(&a, &q).expect("solve_lyapunov should succeed for stable A");

    // Symmetry: X[i,j] == X[j,i].
    for i in 0..2 {
        for j in 0..2 {
            let xij = x[[i, j]];
            let xji = x[[j, i]];
            assert!(
                (xij - xji).abs() < 1e-8,
                "X is not symmetric: X[{i},{j}]={xij} X[{j},{i}]={xji}"
            );
        }
    }

    // Positive definiteness via Sylvester's criterion (leading principal minors > 0).
    let minor1 = x[[0, 0]];
    let minor2 = x[[0, 0]] * x[[1, 1]] - x[[0, 1]] * x[[1, 0]]; // det(X)
    assert!(minor1 > 0.0, "leading 1x1 minor {minor1} is not positive");
    assert!(
        minor2 > 0.0,
        "leading 2x2 minor (det X) {minor2} is not positive"
    );

    // Positive eigenvalues (closed form for a symmetric 2x2 matrix).
    let trace = x[[0, 0]] + x[[1, 1]];
    let det = minor2;
    let disc = (trace * trace - 4.0 * det).max(0.0);
    let sqrt_disc = disc.sqrt();
    let lambda1 = (trace + sqrt_disc) / 2.0;
    let lambda2 = (trace - sqrt_disc) / 2.0;
    assert!(lambda1 > 0.0, "eigenvalue {lambda1} is not positive");
    assert!(lambda2 > 0.0, "eigenvalue {lambda2} is not positive");
}
