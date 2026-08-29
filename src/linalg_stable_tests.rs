//! `#[cfg(test)] mod tests` body for `linalg_stable.rs`, split into this
//! `#[path]` sibling file to keep `linalg_stable.rs` itself under the
//! workspace's 2000-line-per-file limit. Purely mechanical: same module
//! path (`crate::linalg_stable::tests`), same tests, no behavior change.

use super::*;
use approx::assert_relative_eq;

#[test]
fn test_qr_pivoted() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

    let result = StableDecompositions::qr_pivoted(&a).expect("QR pivoted should succeed");

    // Verify dimensions
    assert_eq!(result.q.shape(), vec![2, 2]);
    assert_eq!(result.r.shape(), vec![2, 3]);
    assert_eq!(result.p.shape(), vec![3]);

    // Verify Q is orthogonal (Q^T * Q = I)
    let qt = StableDecompositions::transpose(&result.q).expect("transpose should succeed");
    let qtq = StableDecompositions::matrix_multiply(&qt, &result.q)
        .expect("matrix multiply should succeed");

    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(
                qtq.get(&[i, j]).expect("valid index"),
                expected,
                epsilon = 1e-10
            );
        }
    }
}

#[test]
fn test_cholesky_stable_positive_definite() {
    // Create a positive definite matrix
    let a = Array::from_vec(vec![4.0, 2.0, 2.0, 3.0]).reshape(&[2, 2]);

    let result = StableDecompositions::cholesky_stable(&a).expect("Cholesky should succeed");

    assert!(result.is_positive_definite);
    assert!(!result.pivoting_used);

    // Verify L * L^T = A
    let lt = StableDecompositions::transpose(&result.l).expect("transpose should succeed");
    let llt = StableDecompositions::matrix_multiply(&result.l, &lt)
        .expect("matrix multiply should succeed");

    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(
                llt.get(&[i, j]).expect("valid index"),
                a.get(&[i, j]).expect("valid index"),
                epsilon = 1e-10
            );
        }
    }
}

/// Pre-COW-conversion twin of `ldlt_pivoted`, byte-for-byte identical to
/// the function before the `Array::set` -> bulk-`array_mut()` hoist.
/// `ldlt_pivoted` had (and has) no direct test coverage of its own in
/// this module -- it is only reachable as `cholesky_stable`'s fallback
/// for non-positive-definite input -- so a differential check against
/// this twin is the only available guardrail that the conversion did
/// not change its (pre-existing, possibly numerically quirky) behavior.
fn ldlt_pivoted_precow<T>(a: &Array<T>) -> Result<CholeskyStableResult<T>>
where
    T: Float + Clone + Debug,
{
    let shape = a.shape();
    let n = shape[0];

    let mut l = Array::eye(n, n, 0);
    let mut d = Array::zeros(&[n, n]);
    let mut p: Vec<usize> = (0..n).collect();
    let mut a_work = a.clone();

    for k in 0..n {
        let mut max_val = T::zero();
        let mut pivot_idx = k;

        for i in k..n {
            let abs_val = num_traits::Float::abs(a_work.get(&[i, k])?);
            if abs_val > max_val {
                max_val = abs_val;
                pivot_idx = i;
            }
        }

        if pivot_idx != k {
            for j in 0..n {
                let temp = a_work.get(&[k, j])?;
                a_work.set(&[k, j], a_work.get(&[pivot_idx, j])?)?;
                a_work.set(&[pivot_idx, j], temp)?;

                let temp = a_work.get(&[j, k])?;
                a_work.set(&[j, k], a_work.get(&[j, pivot_idx])?)?;
                a_work.set(&[j, pivot_idx], temp)?;
            }

            p.swap(k, pivot_idx);
        }

        let d_kk = a_work.get(&[k, k])?;
        d.set(&[k, k], d_kk)?;

        if d_kk == T::zero() {
            continue;
        }

        for i in (k + 1)..n {
            let l_ik = a_work.get(&[i, k])? / d_kk;
            l.set(&[i, k], l_ik)?;

            for j in (k + 1)..n {
                let old_val = a_work.get(&[i, j])?;
                let update = l_ik * a_work.get(&[k, j])?;
                a_work.set(&[i, j], old_val - update)?;
            }
        }
    }

    let mut d_min = T::infinity();
    let mut d_max = T::zero();

    for i in 0..n {
        let d_val = num_traits::Float::abs(d.get(&[i, i])?);
        if d_val > T::zero() {
            d_min = d_min.min(d_val);
            d_max = d_max.max(d_val);
        }
    }

    let condition_number = if d_min > T::zero() {
        d_max / d_min
    } else {
        T::infinity()
    };

    Ok(CholeskyStableResult {
        l,
        condition_number,
        is_positive_definite: false,
        pivoting_used: true,
        p: Some(Array::from_vec(p.iter().map(|&x| x as f64).collect())),
        d: Some(d),
    })
}

#[test]
fn test_ldlt_pivoted_matches_precow() {
    // Differential check (not a mathematical-correctness check -- see
    // the doc comment on `ldlt_pivoted_precow`): several indefinite
    // symmetric matrices, each forcing the LDLT fallback path, compared
    // element-for-element between the pre- and post-conversion code.
    let cases: [(&[f64], usize); 3] = [
        (&[1.0, 2.0, 2.0, 1.0], 2),
        (&[0.0, 1.0, 1.0, 0.0], 2),
        (&[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0], 3),
    ];

    for (data, n) in cases {
        let a = Array::from_vec(data.to_vec()).reshape(&[n, n]);

        let expected = ldlt_pivoted_precow(&a).expect("precow LDLT should succeed");
        let actual = StableDecompositions::ldlt_pivoted(&a).expect("converted LDLT should succeed");

        assert_eq!(actual.is_positive_definite, expected.is_positive_definite);
        assert_eq!(actual.pivoting_used, expected.pivoting_used);
        assert_relative_eq!(
            actual.condition_number,
            expected.condition_number,
            epsilon = 1e-12
        );

        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(
                    actual.l.get(&[i, j]).expect("valid index"),
                    expected.l.get(&[i, j]).expect("valid index"),
                    epsilon = 1e-12
                );
            }
        }

        let actual_d = actual.d.expect("D present");
        let expected_d = expected.d.expect("D present");
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(
                    actual_d.get(&[i, j]).expect("valid index"),
                    expected_d.get(&[i, j]).expect("valid index"),
                    epsilon = 1e-12
                );
            }
        }

        let actual_p = actual.p.expect("P present").to_vec();
        let expected_p = expected.p.expect("P present").to_vec();
        assert_eq!(actual_p, expected_p);
    }
}

#[test]
fn test_symmetric_eigendecomposition_2x2() {
    let a = Array::from_vec(vec![3.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);

    let (eigenvalues, eigenvectors) = StableDecompositions::symmetric_eigendecomposition(&a)
        .expect("eigendecomposition should succeed");

    assert_eq!(eigenvalues.len(), 2);
    assert_eq!(eigenvectors.shape(), vec![2, 2]);

    // For this matrix, eigenvalues should be 4 and 2
    let mut sorted_eigenvalues = eigenvalues.clone();
    sorted_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    assert_relative_eq!(sorted_eigenvalues[0], 4.0, epsilon = 1e-10);
    assert_relative_eq!(sorted_eigenvalues[1], 2.0, epsilon = 1e-10);
}

#[test]
fn test_svd_stable_small() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

    let result = StableDecompositions::svd_stable(&a).expect("SVD should succeed");

    // Verify dimensions
    assert_eq!(result.u.shape(), vec![2, 2]);
    assert_eq!(result.s.shape(), vec![2]);
    assert_eq!(result.vt.shape(), vec![2, 2]);

    // Verify singular values are non-negative and sorted
    let s_data = result.s.to_vec();
    assert!(s_data[0] >= s_data[1]);
    assert!(s_data[1] >= 0.0);
}

#[test]
fn test_householder_vector() {
    let x = vec![1.0, 2.0, 3.0];
    let (v, beta) =
        StableDecompositions::householder_vector(&x).expect("householder should succeed");

    assert_eq!(v.len(), 3);
    assert!(beta >= 0.0);

    // Verify that applying the Householder reflection gives correct result
    let result = StableDecompositions::apply_householder(&x, &v, beta)
        .expect("apply householder should succeed");

    // First component should have the opposite sign and same magnitude as original norm
    let x_norm = (1.0 + 4.0 + 9.0_f64).sqrt();
    assert_relative_eq!(result[0].abs(), x_norm, epsilon = 1e-10);

    // Other components should be zero
    assert_relative_eq!(result[1], 0.0, epsilon = 1e-10);
    assert_relative_eq!(result[2], 0.0, epsilon = 1e-10);
}

// -------------------------------------------------------------------------
// Tests for the new full-size implementations
// -------------------------------------------------------------------------

/// Helper: compute ||A·v - λ·v||₂ for an eigenpair.
fn eigenpair_residual(a: &Array<f64>, lambda: f64, v: &[f64]) -> f64 {
    let n = v.len();
    let mut res = 0.0_f64;
    for i in 0..n {
        let mut av_i = 0.0_f64;
        for j in 0..n {
            av_i += a.get(&[i, j]).expect("valid index") * v[j];
        }
        let diff = av_i - lambda * v[i];
        res += diff * diff;
    }
    res.sqrt()
}

/// Helper: Frobenius norm ||A||_F.
fn frob_norm(a: &Array<f64>, rows: usize, cols: usize) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let v = a.get(&[i, j]).expect("valid index");
            s += v * v;
        }
    }
    s.sqrt()
}

#[test]
fn test_symmetric_eigen_4x4() {
    // Build a known real symmetric 4×4 matrix:
    //   A = Lᵀ·L where L is lower triangular, so A is positive definite.
    // Values chosen so eigenvalues are well-separated.
    #[rustfmt::skip]
    let data: Vec<f64> = vec![
         4.0, 2.0, 1.0, 0.5,
         2.0, 5.0, 2.0, 1.0,
         1.0, 2.0, 6.0, 2.0,
         0.5, 1.0, 2.0, 7.0,
    ];
    let a = Array::from_vec(data).reshape(&[4, 4]);

    let (eigenvalues, eigenvectors) = StableDecompositions::symmetric_eigendecomposition(&a)
        .expect("eigendecomposition should succeed");

    let n = 4;
    assert_eq!(eigenvalues.len(), n);
    assert_eq!(eigenvectors.shape(), vec![n, n]);

    // Verify Av = λv  and  |vᵢᵀ·vⱼ| ≈ δᵢⱼ
    for col in 0..n {
        let lambda = eigenvalues[col];
        let v: Vec<f64> = (0..n)
            .map(|r| eigenvectors.get(&[r, col]).expect("valid"))
            .collect();

        // Residual ||Av - λv|| < 1e-6
        let res = eigenpair_residual(&a, lambda, &v);
        assert!(
            res < 1e-6,
            "eigenpair residual {} for eigenvalue {} is too large",
            res,
            lambda
        );

        // Orthogonality of distinct eigenvectors
        for col2 in (col + 1)..n {
            let v2: Vec<f64> = (0..n)
                .map(|r| eigenvectors.get(&[r, col2]).expect("valid"))
                .collect();
            let dot: f64 = v.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
            assert!(
                dot.abs() < 1e-6,
                "eigenvectors {} and {} not orthogonal: dot = {}",
                col,
                col2,
                dot
            );
        }
    }
}

#[test]
fn test_symmetric_eigen_5x5() {
    // 5×5 symmetric matrix: Hilbert-like, known to be positive definite.
    let n = 5usize;
    let mut data = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = 1.0 / (1.0 + (i + j) as f64);
        }
    }
    // Make strictly diagonally dominant to guarantee positive definiteness.
    for i in 0..n {
        let mut row_sum = 0.0_f64;
        for j in 0..n {
            if j != i {
                row_sum += data[i * n + j].abs();
            }
        }
        data[i * n + i] = row_sum + 2.0;
    }
    let a = Array::from_vec(data).reshape(&[n, n]);

    let (eigenvalues, eigenvectors) = StableDecompositions::symmetric_eigendecomposition(&a)
        .expect("eigendecomposition should succeed");

    assert_eq!(eigenvalues.len(), n);

    for col in 0..n {
        let lambda = eigenvalues[col];
        let v: Vec<f64> = (0..n)
            .map(|r| eigenvectors.get(&[r, col]).expect("valid"))
            .collect();

        let res = eigenpair_residual(&a, lambda, &v);
        assert!(
            res < 1e-5,
            "5×5 eigenpair residual {} for eigenvalue {} is too large",
            res,
            lambda
        );

        for col2 in (col + 1)..n {
            let v2: Vec<f64> = (0..n)
                .map(|r| eigenvectors.get(&[r, col2]).expect("valid"))
                .collect();
            let dot: f64 = v.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
            assert!(
                dot.abs() < 1e-5,
                "5×5 eigenvectors {} and {} not orthogonal: dot = {}",
                col,
                col2,
                dot
            );
        }
    }
}

#[test]
fn test_svd_bidiagonal_4x4() {
    // 4×4 non-symmetric matrix (thin, well-conditioned).
    #[rustfmt::skip]
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
    ];
    let a = Array::from_vec(data).reshape(&[4, 4]);

    // svd_stable dispatches to svd_bidiagonal for min_mn > 4;
    // for 4×4 it goes to svd_small_stable (min_mn==4 uses the <=4 path).
    // We call svd_bidiagonal directly to test the implementation.
    let result = StableDecompositions::svd_bidiagonal(&a).expect("SVD bidiagonal should succeed");

    let m = 4usize;
    let n = 4usize;
    let k = m.min(n);

    assert_eq!(result.u.shape(), vec![m, k]);
    assert_eq!(result.s.shape(), vec![k]);
    assert_eq!(result.vt.shape(), vec![k, n]);

    // Verify A ≈ U · diag(σ) · Vᵀ
    let mut recon = Array::zeros(&[m, n]);
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0_f64;
            for l in 0..k {
                val += result.u.get(&[i, l]).expect("u")
                    * result.s.get(&[l]).expect("s")
                    * result.vt.get(&[l, j]).expect("vt");
            }
            recon.set(&[i, j], val).expect("set");
        }
    }
    let err = frob_norm(&recon, m, n) + {
        // compute || A - recon ||_F
        let mut diff_sq = 0.0_f64;
        for i in 0..m {
            for j in 0..n {
                let d = a.get(&[i, j]).expect("a") - recon.get(&[i, j]).expect("recon");
                diff_sq += d * d;
            }
        }
        diff_sq.sqrt() - frob_norm(&recon, m, n)
    };
    // Just compute the Frobenius norm of the difference directly.
    let mut diff_frob = 0.0_f64;
    for i in 0..m {
        for j in 0..n {
            let d = a.get(&[i, j]).expect("a") - recon.get(&[i, j]).expect("recon");
            diff_frob += d * d;
        }
    }
    diff_frob = diff_frob.sqrt();
    let _ = err;
    assert!(
        diff_frob < 1e-6,
        "SVD 4×4 reconstruction error ||A - UΣVᵀ||_F = {} is too large",
        diff_frob
    );

    // U orthogonality: Uᵀ·U ≈ I_k
    for i in 0..k {
        for j in 0..k {
            let mut dot = 0.0_f64;
            for r in 0..m {
                dot += result.u.get(&[r, i]).expect("u") * result.u.get(&[r, j]).expect("u");
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-6,
                "U not orthogonal at ({},{}) = {}",
                i,
                j,
                dot
            );
        }
    }

    // Vᵀ orthogonality: Vᵀ·V = Vᵀ·(Vᵀ)ᵀ ≈ I_k
    for i in 0..k {
        for j in 0..k {
            let mut dot = 0.0_f64;
            for c in 0..n {
                dot += result.vt.get(&[i, c]).expect("vt") * result.vt.get(&[j, c]).expect("vt");
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-6,
                "Vᵀ rows not orthogonal at ({},{}) = {}",
                i,
                j,
                dot
            );
        }
    }
}

#[test]
fn test_svd_bidiagonal_3x5() {
    // 3×5 rectangular matrix.
    #[rustfmt::skip]
    let data: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 1.0,
        2.0, 3.0, 4.0, 5.0, 6.0,
    ];
    let a = Array::from_vec(data).reshape(&[3, 5]);

    let result =
        StableDecompositions::svd_bidiagonal(&a).expect("SVD bidiagonal 3x5 should succeed");

    let m = 3usize;
    let n = 5usize;
    let k = m.min(n); // = 3

    assert_eq!(result.u.shape(), vec![m, k]);
    assert_eq!(result.s.shape(), vec![k]);
    assert_eq!(result.vt.shape(), vec![k, n]);

    // Verify A ≈ U · diag(σ) · Vᵀ
    let mut diff_frob = 0.0_f64;
    for i in 0..m {
        for j in 0..n {
            let mut val = 0.0_f64;
            for l in 0..k {
                val += result.u.get(&[i, l]).expect("u")
                    * result.s.get(&[l]).expect("s")
                    * result.vt.get(&[l, j]).expect("vt");
            }
            let d = a.get(&[i, j]).expect("a") - val;
            diff_frob += d * d;
        }
    }
    diff_frob = diff_frob.sqrt();
    assert!(
        diff_frob < 1e-6,
        "SVD 3×5 reconstruction error ||A - UΣVᵀ||_F = {} is too large",
        diff_frob
    );

    // U orthogonality
    for i in 0..k {
        for j in 0..k {
            let mut dot = 0.0_f64;
            for r in 0..m {
                dot += result.u.get(&[r, i]).expect("u") * result.u.get(&[r, j]).expect("u");
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-6,
                "3×5 U not orthogonal at ({},{}) = {}",
                i,
                j,
                dot
            );
        }
    }

    // Vᵀ row orthogonality
    for i in 0..k {
        for j in 0..k {
            let mut dot = 0.0_f64;
            for c in 0..n {
                dot += result.vt.get(&[i, c]).expect("vt") * result.vt.get(&[j, c]).expect("vt");
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-6,
                "3×5 Vᵀ rows not orthogonal at ({},{}) = {}",
                i,
                j,
                dot
            );
        }
    }
}
