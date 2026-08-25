//! Reference tests for linear algebra operations
//!
//! This file tests NumRS2's linear algebra operations against known reference values
//! to ensure correctness and numerical stability.
//!
//! Note: These tests require the 'lapack' feature to be enabled.

#![cfg(feature = "lapack")]
#![allow(deprecated)] // Suppress deprecation warnings for transitional modules
#![allow(clippy::result_large_err)]

use approx::{assert_abs_diff_eq, assert_relative_eq};
use num_traits::sign::Signed;
use numrs2::prelude::*;

// Import from the core linalg module
use numrs2::linalg::matrix_ops::det;
use numrs2::linalg::solve::{inv, solve};
use numrs2::linalg::vector_ops::{norm, trace};

#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
use numrs2::linalg::decomposition::{cholesky, qr, svd};
use numrs2::new_modules::matrix_decomp::condition_number;
#[cfg(feature = "matrix_decomp")]
use numrs2::new_modules::matrix_decomp::lu;

// Import additional functions that may be feature-gated
use numrs2::linalg::decomposition::matrix_rank;

// Use SciRS2 functions when available
#[cfg(feature = "scirs")]
use scirs2_linalg::{eigh as scirs_eigh, schur as scirs_schur};

// `matrix_power` is implemented locally in numrs2 itself (binary
// exponentiation, see `linalg::matrix_ops::matrix_power`) rather than
// delegated to `scirs2_linalg::matrix_power`, which is limited to `|n| <= 1`
// (that limitation is what `test_matrix_power_reference` below used to be
// `#[ignore]`d for). This file is `#![cfg(feature = "lapack")]`-gated as a
// whole, and `numrs2::linalg::matrix_power` requires exactly that feature,
// so it is always available here -- no `scirs`/`not(scirs)` split needed.
fn matrix_power(a: &Array<f64>, n: i32) -> numrs2::error::Result<Array<f64>> {
    numrs2::linalg::matrix_power(a, n)
}

#[cfg(feature = "scirs")]
fn schur(a: &Array<f64>) -> numrs2::error::Result<(Array<f64>, Array<f64>)> {
    // Convert numrs2 Array to ndarray ArrayView2 for scirs2
    let a_view = a.view_2d().map_err(|e| {
        numrs2::error::NumRs2Error::ComputationError(format!("View conversion failed: {:?}", e))
    })?;
    let (q, t) = scirs_schur(&a_view).map_err(|e| {
        numrs2::error::NumRs2Error::ComputationError(format!("SCIRS schur failed: {:?}", e))
    })?;

    // Convert back to numrs2 Arrays
    let q_converted = Array::from_ndarray(q.into_dyn());
    let t_converted = Array::from_ndarray(t.into_dyn());

    Ok((q_converted, t_converted))
}

// Provide fallback implementations for missing functions
#[cfg(not(feature = "scirs"))]
fn schur(a: &Array<f64>) -> numrs2::error::Result<(Array<f64>, Array<f64>)> {
    // Use NumRS2's own schur implementation
    #[cfg(feature = "matrix_decomp")]
    {
        numrs2::new_modules::matrix_decomp::schur(a)
    }
    #[cfg(not(feature = "matrix_decomp"))]
    {
        Err(numrs2::error::NumRs2Error::FeatureNotEnabled(
            "matrix_decomp feature required for schur".to_string(),
        ))
    }
}

// Unified eigh function that works with different feature configurations
#[cfg(feature = "scirs")]
fn eigh(a: &Array<f64>, _uplo: &str) -> numrs2::error::Result<(Array<f64>, Array<f64>)> {
    // Convert numrs2 Array to ndarray ArrayView2 for scirs2
    let a_view = a.view_2d().map_err(|e| {
        numrs2::error::NumRs2Error::ComputationError(format!("View conversion failed: {:?}", e))
    })?;
    let (vals, vecs) = scirs_eigh(&a_view, None).map_err(|e| {
        numrs2::error::NumRs2Error::ComputationError(format!("SCIRS eigh failed: {:?}", e))
    })?;

    // Convert back to numrs2 Arrays
    let eigenvalues_converted = Array::from_ndarray(vals.into_dyn());
    let eigenvectors_converted = Array::from_ndarray(vecs.into_dyn());

    Ok((eigenvalues_converted, eigenvectors_converted))
}

#[cfg(not(feature = "scirs"))]
fn eigh(_a: &Array<f64>, _uplo: &str) -> numrs2::error::Result<(Array<f64>, Array<f64>)> {
    Err(numrs2::error::NumRs2Error::FeatureNotEnabled(
        "scirs or matrix_decomp feature required for eigh".to_string(),
    ))
}

// Tolerance for floating point comparisons
const TOLERANCE: f64 = 1e-10;

/// Helper function to check if a value is within expected range
fn is_within_range(value: f64, expected: f64, tolerance: f64) -> bool {
    (value - expected).abs() <= tolerance
}

/// Helper function to create a test matrix with known properties
fn create_test_matrix() -> Array<f64> {
    // 3x3 matrix with known determinant, eigenvalues, etc.
    // [ 4  1  1 ]
    // [ 1  3  1 ]
    // [ 1  1  2 ]
    let mut m = Array::<f64>::zeros(&[3, 3]);
    m.set(&[0, 0], 4.0).unwrap();
    m.set(&[0, 1], 1.0).unwrap();
    m.set(&[0, 2], 1.0).unwrap();
    m.set(&[1, 0], 1.0).unwrap();
    m.set(&[1, 1], 3.0).unwrap();
    m.set(&[1, 2], 1.0).unwrap();
    m.set(&[2, 0], 1.0).unwrap();
    m.set(&[2, 1], 1.0).unwrap();
    m.set(&[2, 2], 2.0).unwrap();
    m
}

/// Helper function to create a known square matrix for testing
fn create_known_square_matrix() -> Array<f64> {
    // [ 1  2  3 ]
    // [ 4  5  6 ]
    // [ 7  8  9 ]
    let mut m = Array::<f64>::zeros(&[3, 3]);
    m.set(&[0, 0], 1.0).unwrap();
    m.set(&[0, 1], 2.0).unwrap();
    m.set(&[0, 2], 3.0).unwrap();
    m.set(&[1, 0], 4.0).unwrap();
    m.set(&[1, 1], 5.0).unwrap();
    m.set(&[1, 2], 6.0).unwrap();
    m.set(&[2, 0], 7.0).unwrap();
    m.set(&[2, 1], 8.0).unwrap();
    m.set(&[2, 2], 9.0).unwrap();
    m
}

/// Helper function to create a rectangle matrix for testing
fn create_rectangle_matrix() -> Array<f64> {
    // [ 1  2  3 ]
    // [ 4  5  6 ]
    let mut m = Array::<f64>::zeros(&[2, 3]);
    m.set(&[0, 0], 1.0).unwrap();
    m.set(&[0, 1], 2.0).unwrap();
    m.set(&[0, 2], 3.0).unwrap();
    m.set(&[1, 0], 4.0).unwrap();
    m.set(&[1, 1], 5.0).unwrap();
    m.set(&[1, 2], 6.0).unwrap();
    m
}

#[test]
fn test_matmul_reference() {
    // Test matrix multiplication against known result
    let a = create_known_square_matrix();
    let b = create_known_square_matrix();

    // Expected result of [ 1  2  3 ] * [ 1  2  3 ] = [ 30  36  42 ]
    //                    [ 4  5  6 ]   [ 4  5  6 ]   [ 66  81  96 ]
    //                    [ 7  8  9 ]   [ 7  8  9 ]   [102 126 150 ]
    let expected_values = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0];

    let c = a.matmul(&b).unwrap();

    // Check each value
    let c_vec = c.to_vec();
    for (actual, expected) in c_vec.iter().zip(expected_values.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = TOLERANCE);
    }

    // Test matrix-vector multiplication
    let v = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0]);

    // Expected result of [ 1  2  3 ] * [ 1 ] = [ 14 ]
    //                    [ 4  5  6 ]   [ 2 ]   [ 32 ]
    //                    [ 7  8  9 ]   [ 3 ]   [ 50 ]
    let expected_values = [14.0, 32.0, 50.0];

    let result = a.matmul(&v.reshape(&[3, 1])).unwrap().reshape(&[3]);

    // Check each value
    let result_vec = result.to_vec();
    for (actual, expected) in result_vec.iter().zip(expected_values.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = TOLERANCE);
    }
}

#[test]
fn test_determinant_reference() {
    // Test determinant against known value

    // Determinant of the test matrix should be 17
    let m = create_test_matrix();
    let det_m = det(&m).unwrap();
    assert_relative_eq!(det_m, 17.0, epsilon = TOLERANCE);

    // Determinant of [ 1  2  3 ]
    //                [ 4  5  6 ] is 0 (singular matrix)
    //                [ 7  8  9 ]
    let singular = create_known_square_matrix();
    let det_singular = det(&singular).unwrap();
    assert_abs_diff_eq!(det_singular, 0.0, epsilon = TOLERANCE);

    // Determinant of identity matrix is 1
    let identity = Array::<f64>::eye(3, 3, 0);
    let det_identity = det(&identity).unwrap();
    assert_relative_eq!(det_identity, 1.0, epsilon = TOLERANCE);
}

#[test]
fn test_inverse_reference() {
    // Test matrix inverse against known value
    let m = create_test_matrix();
    let m_inv = inv(&m).unwrap();

    // Expected inverse of the test matrix (computed accurately)
    // [ 0.29411764705882354 -0.05882353 -0.11764706 ]
    // [-0.05882353  0.41176471 -0.17647059 ]
    // [-0.11764706 -0.17647059  0.64705882 ]
    let expected_values = [
        0.29411764705882354,
        -0.058823529411764705,
        -0.11764705882352941,
        -0.058823529411764705,
        0.4117647058823529,
        -0.1764705882352941,
        -0.11764705882352941,
        -0.1764705882352941,
        0.6470588235294118,
    ];

    // Check each value
    let m_inv_vec = m_inv.to_vec();
    for (actual, expected) in m_inv_vec.iter().zip(expected_values.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = TOLERANCE);
    }

    // Test that A * A^-1 = I
    let product = m.matmul(&m_inv).unwrap();

    // Check that the product is approximately the identity matrix
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(product.get(&[i, j]).unwrap(), expected, epsilon = TOLERANCE);
        }
    }
}

#[test]
fn test_eigendecomposition_reference() {
    // Was `#[ignore]`d as "Eigenvalue computation differences between
    // implementations": the previous body only ever compared eigen*values*
    // (the eigenvectors were discarded via `_`), but different backends are
    // free to return eigenvectors -- and even same-magnitude eigenvalues --
    // in different orders, and eigenvectors are only defined up to sign (and,
    // for repeated eigenvalues, up to an arbitrary rotation within the
    // eigenspace). Comparing raw eigenvector elements against one backend's
    // specific choice is therefore not a portable cross-implementation check.
    //
    // Rewritten to check the two properties that any correct symmetric
    // eigendecomposition must satisfy, regardless of backend conventions:
    //   1. The eigenvalue *multiset* matches the reference values (order
    //      doesn't matter -- both sides are sorted before comparing).
    //   2. Every returned eigenpair actually satisfies `A v = lambda v`
    //      (the defining residual) with `v` unit-norm.
    let m = create_test_matrix();
    let n = 3;

    let (eigenvalues, eigenvectors) = eigh(&m, "lower").unwrap();

    // Reference eigenvalues for `create_test_matrix()` ([[4,1,1],[1,3,1],[1,1,2]]):
    // trace = 9 and det = 17 both check out against these three values.
    let mut eigenvalues_sorted = eigenvalues.to_vec();
    eigenvalues_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let expected_desc = [5.214319743377534_f64, 2.4608111271891095, 1.324869129433354];
    for (got, want) in eigenvalues_sorted.iter().zip(expected_desc.iter()) {
        assert_relative_eq!(*got, *want, epsilon = TOLERANCE);
    }

    // Residual check: pair each eigenvalue with its *own* column (backend
    // order, not the sorted-for-comparison order above) and verify
    // `A v_k - lambda_k v_k ~= 0`, plus that `v_k` is unit-norm.
    const RESIDUAL_TOL: f64 = 1e-8;
    for k in 0..n {
        let lambda = eigenvalues.get(&[k]).unwrap();
        let v: Vec<f64> = (0..n).map(|i| eigenvectors.get(&[i, k]).unwrap()).collect();

        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < RESIDUAL_TOL,
            "eigenvector {k} should be unit-norm, got ||v||^2 = {norm_sq}"
        );

        for i in 0..n {
            let av_i: f64 = (0..n).map(|j| m.get(&[i, j]).unwrap() * v[j]).sum();
            let residual = av_i - lambda * v[i];
            assert!(
                residual.abs() < RESIDUAL_TOL,
                "eigenpair {k} (lambda={lambda}): (A v - lambda v)[{i}] = {residual}, expected ~0"
            );
        }
    }
}

#[test]
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
#[allow(deprecated)]
fn test_svd_reference() {
    // Test SVD against known values for a simple matrix
    let m = create_rectangle_matrix();

    // The singular values of the 2x3 matrix
    // [ 1  2  3 ]
    // [ 4  5  6 ]
    // are approximately 9.508032 and 0.77286964
    let (_, s, _) = svd(&m).unwrap();

    // Extract the diagonal values (singular values) from the S matrix
    let s_diag = if s.shape().len() == 2 {
        // S is a diagonal matrix, extract diagonal
        let min_dim = s.shape()[0].min(s.shape()[1]);
        let mut singular_values = Vec::new();
        for i in 0..min_dim {
            if let Ok(val) = s.get(&[i, i]) {
                if val.abs() > 1e-10 {
                    // Only include non-zero values
                    singular_values.push(val);
                }
            }
        }
        singular_values
    } else {
        // S is already a vector of singular values
        s.to_vec()
    };

    // Check that we have the right number of singular values
    assert_eq!(s_diag.len(), 2);

    // Check against expected values (within tolerance)
    assert!(is_within_range(s_diag[0], 9.508032, 0.01));
    assert!(is_within_range(s_diag[1], 0.77286964, 0.01));
}

#[test]
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
#[allow(deprecated)]
fn test_qr_decomposition_reference() {
    // Test QR decomposition with a matrix that has known factors
    let m = Array::<f64>::from_vec(vec![12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0])
        .reshape(&[3, 3]);

    println!("Input matrix m: {:?}", m.to_vec());
    let (q, r) = qr(&m).unwrap();
    println!("Q matrix: {:?}", q.to_vec());
    println!("R matrix: {:?}", r.to_vec());

    // Expected values for Q (approximate due to potential sign differences)
    let expected_q_abs = [
        6.0 / 7.0,
        -69.0 / 175.0,
        -58.0 / 175.0,
        3.0 / 7.0,
        158.0 / 175.0,
        6.0 / 175.0,
        -2.0 / 7.0,
        6.0 / 35.0,
        -33.0 / 35.0,
    ];

    // Check Q's orthogonality
    let q_t = q.transpose();
    let q_t_q = q_t.matmul(&q).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(q_t_q.get(&[i, j]).unwrap(), expected, epsilon = TOLERANCE);
        }
    }

    // Check Q's components (absolute values to account for possible sign differences)
    let q_vec = q.to_vec();
    for (actual, expected) in q_vec.iter().zip(expected_q_abs.iter()) {
        assert_relative_eq!(actual.abs(), expected.abs(), epsilon = 0.01);
    }

    // Check R is upper triangular
    for i in 0..3 {
        for j in 0..i {
            assert_relative_eq!(r.get(&[i, j]).unwrap(), 0.0, epsilon = TOLERANCE);
        }
    }

    // Verify A = Q * R
    let qr = q.matmul(&r).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            assert_relative_eq!(
                qr.get(&[i, j]).unwrap(),
                m.get(&[i, j]).unwrap(),
                epsilon = TOLERANCE
            );
        }
    }
}

#[test]
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
#[allow(deprecated)]
fn test_cholesky_decomposition_reference() {
    // Test Cholesky decomposition with a matrix that has a known factor
    // Create a positive definite matrix with known Cholesky decomposition
    // A = L * L^T where L is known

    // Create L:
    // [ 2  0  0 ]
    // [ 1  2  0 ]
    // [ 1  3  1 ]
    let mut l = Array::<f64>::zeros(&[3, 3]);
    l.set(&[0, 0], 2.0).unwrap();
    l.set(&[1, 0], 1.0).unwrap();
    l.set(&[1, 1], 2.0).unwrap();
    l.set(&[2, 0], 1.0).unwrap();
    l.set(&[2, 1], 3.0).unwrap();
    l.set(&[2, 2], 1.0).unwrap();

    // Create A = L * L^T
    let l_t = l.transpose();
    let a = l.matmul(&l_t).unwrap();

    // Compute Cholesky decomposition
    let l_computed = cholesky(&a).unwrap();

    // Check against expected values
    let l_vec = l.to_vec();
    let l_computed_vec = l_computed.to_vec();

    for (actual, expected) in l_computed_vec.iter().zip(l_vec.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = TOLERANCE);
    }
}

#[cfg(feature = "matrix_decomp")]
#[test]
fn test_lu_decomposition_reference() {
    // Test LU decomposition with a matrix that has known factors
    // Create a matrix with known LU decomposition
    let m = Array::<f64>::from_vec(vec![2.0, 1.0, 1.0, 4.0, 10.0, -1.0, 3.0, 5.0, 0.0])
        .reshape(&[3, 3]);

    // Compute LU decomposition
    #[allow(deprecated)]
    let (l, _u, _p) = lu(&m).unwrap();

    // Check L is lower triangular - different LU implementations have different forms
    // Some implementations return LDU where diagonal is merged into L or U
    // The key property is that the reconstruction L*U = A works correctly
    for i in 0..3 {
        for j in 0..3 {
            if j > i {
                // Upper triangle should be zero for L matrix
                let val = l.get(&[i, j]).unwrap();
                assert!(val.abs() <= TOLERANCE, "L should be lower triangular");
            }
        }
    }

    // Verify the reconstruction L*U = A
    let reconstructed = l.matmul(&_u).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let orig = m.get(&[i, j]).unwrap();
            let recon = reconstructed.get(&[i, j]).unwrap();
            assert_relative_eq!(orig, recon, epsilon = TOLERANCE);
        }
    }

    // LU decomposition properties verified
}

#[test]
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
#[allow(deprecated)]
fn test_norm_reference() {
    // Test matrix norms against known values

    // Create a matrix with known norms
    let m = Array::<f64>::from_vec(vec![3.0, 4.0, 0.0, 0.0]).reshape(&[2, 2]);

    // Frobenius norm should be 5 (Euclidean norm of all elements)
    let frob_norm = norm(&m, Some(2.0)).unwrap();
    assert_relative_eq!(frob_norm, 5.0, epsilon = TOLERANCE);

    // Nuclear norm (sum of singular values)
    let (_, s, _) = svd(&m).unwrap();
    let nuclear_norm = s.sum();
    // Expected value is 5
    assert_relative_eq!(nuclear_norm, 5.0, epsilon = TOLERANCE);

    // Test with vector
    let v = Array::<f64>::from_vec(vec![3.0, 4.0]);

    // L1 norm (sum of absolute values) = 7
    let l1_norm = norm(&v, Some(1.0)).unwrap();
    assert_relative_eq!(l1_norm, 7.0, epsilon = TOLERANCE);

    // L2 norm (Euclidean norm) = 5
    let l2_norm = norm(&v, Some(2.0)).unwrap();
    assert_relative_eq!(l2_norm, 5.0, epsilon = TOLERANCE);

    // L-infinity norm (maximum absolute value) = 4
    let inf_norm = norm(&v, Some(f64::INFINITY)).unwrap();
    assert_relative_eq!(inf_norm, 4.0, epsilon = TOLERANCE);
}

#[test]
#[cfg(all(feature = "matrix_decomp", feature = "lapack"))]
#[allow(deprecated)]
fn test_norm_ord_neg2_and_nuclear_reference() {
    use numrs2::linalg::nuclear_norm;

    // Diagonal matrix: singular values are the absolute values of the
    // diagonal entries, {3, 4}.
    let m = Array::<f64>::from_vec(vec![3.0, 0.0, 0.0, -4.0]).reshape(&[2, 2]);

    // np.linalg.norm(m, ord=-2) == 3.0 (smallest singular value)
    let smallest_sv = norm(&m, Some(-2.0)).unwrap();
    assert_relative_eq!(smallest_sv, 3.0, epsilon = TOLERANCE);

    // np.linalg.norm(m, ord='nuc') == 7.0 (sum of singular values)
    let nuc = nuclear_norm(&m).unwrap();
    assert_relative_eq!(nuc, 7.0, epsilon = TOLERANCE);

    // Non-square matrix, cross-checked against numpy.linalg.norm with
    // ord=-2/'nuc' (numpy 2.4.2): B = [[1,2],[3,4],[5,6]], singular values
    // ~= [9.52551809156511, 0.5143005806586446].
    let b = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[3, 2]);
    let b_smallest_sv = norm(&b, Some(-2.0)).unwrap();
    assert_relative_eq!(b_smallest_sv, 0.5143005806586446, epsilon = 1e-9);
    let b_nuc = nuclear_norm(&b).unwrap();
    assert_relative_eq!(b_nuc, 9.52551809156511 + 0.5143005806586446, epsilon = 1e-8);
}

#[test]
fn test_trace_reference() {
    // Test trace against known values

    // Create a matrix with known trace
    let m =
        Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(&[3, 3]);

    // Trace should be 1 + 5 + 9 = 15
    let tr = trace(&m).unwrap();
    assert_relative_eq!(tr, 15.0, epsilon = TOLERANCE);

    // Trace of identity should be its dimension
    let identity = Array::<f64>::eye(5, 5, 0);
    let tr_identity = trace(&identity).unwrap();
    assert_relative_eq!(tr_identity, 5.0, epsilon = TOLERANCE);

    // Trace of zero matrix should be 0
    let zero = Array::<f64>::zeros(&[4, 4]);
    let tr_zero = trace(&zero).unwrap();
    assert_relative_eq!(tr_zero, 0.0, epsilon = TOLERANCE);
}

#[test]
fn test_solve_reference() {
    // Test solving linear systems against known values

    // Create a system with known solution
    // [ 2  1 ] [ x ] = [ 5 ]
    // [ 1  3 ] [ y ]   [ 7 ]
    let a = Array::<f64>::from_vec(vec![2.0, 1.0, 1.0, 3.0]).reshape(&[2, 2]);
    let b = Array::<f64>::from_vec(vec![5.0, 7.0]);

    // Expected solution: x = 1.6, y = 1.8
    let x = solve(&a, &b).unwrap();

    assert_relative_eq!(x.get(&[0]).unwrap(), 1.6, epsilon = TOLERANCE);
    assert_relative_eq!(x.get(&[1]).unwrap(), 1.8, epsilon = TOLERANCE);

    // Verify: A * x = b
    let ax = a.matmul(&x.reshape(&[2, 1])).unwrap().reshape(&[2]);

    assert_relative_eq!(
        ax.get(&[0]).unwrap(),
        b.get(&[0]).unwrap(),
        epsilon = TOLERANCE
    );
    assert_relative_eq!(
        ax.get(&[1]).unwrap(),
        b.get(&[1]).unwrap(),
        epsilon = TOLERANCE
    );
}

#[cfg(feature = "matrix_decomp")]
#[test]
fn test_rank_reference() {
    // Test matrix rank against known values

    // Full rank matrix (rank = 3)
    let full_rank = create_test_matrix();
    let rank_val = matrix_rank(&full_rank, None).unwrap();

    assert_eq!(rank_val, 3);

    // Rank deficient matrix (rank = 2)
    let singular = create_known_square_matrix();
    let singular_rank = matrix_rank(&singular, None).unwrap();
    assert_eq!(singular_rank, 2);

    // Rank 1 matrix
    let mut rank1 = Array::<f64>::zeros(&[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            rank1
                .set(&[i, j], (i as f64 + 1.0) * (j as f64 + 1.0))
                .unwrap();
        }
    }
    let rank1_val = matrix_rank(&rank1, None).unwrap();
    assert_eq!(rank1_val, 1);

    // Zero matrix (rank = 0)
    let zero = Array::<f64>::zeros(&[3, 3]);
    let zero_rank = matrix_rank(&zero, None).unwrap();
    assert_eq!(zero_rank, 0);
}

#[cfg(feature = "matrix_decomp")]
#[test]
fn test_condition_number_reference() {
    // Test condition number against known values

    // Identity matrix should have condition number 1
    let identity = Array::<f64>::eye(3, 3, 0);
    #[allow(deprecated)]
    let cond_identity = condition_number(&identity).unwrap();
    assert_relative_eq!(cond_identity, 1.0, epsilon = TOLERANCE);

    // Symmetric matrix with eigenvalues 3, 2, 1 has condition number 3/1 = 3
    let mut symmetric = Array::<f64>::zeros(&[3, 3]);
    symmetric.set(&[0, 0], 3.0).unwrap();
    symmetric.set(&[1, 1], 2.0).unwrap();
    symmetric.set(&[2, 2], 1.0).unwrap();

    #[allow(deprecated)]
    let cond_symmetric = condition_number(&symmetric).unwrap();
    assert_relative_eq!(cond_symmetric, 3.0, epsilon = TOLERANCE);

    // Nearly singular matrix
    let mut nearly_singular = Array::<f64>::eye(3, 3, 0);
    nearly_singular.set(&[0, 0], 1000.0).unwrap();
    nearly_singular.set(&[2, 2], 0.001).unwrap();

    #[allow(deprecated)]
    let cond_nearly_singular = condition_number(&nearly_singular).unwrap();
    assert_relative_eq!(cond_nearly_singular, 1000000.0, epsilon = 0.01);
}

#[test]
fn test_matrix_power_reference() {
    // Test matrix power against known values. `matrix_power` above now
    // delegates to numrs2's own binary-exponentiation implementation
    // instead of `scirs2_linalg::matrix_power` (which was limited to
    // `|n| <= 1`), so this no longer needs `#[ignore]`.

    // Create a test matrix
    let m = Array::<f64>::from_vec(vec![1.0, 1.0, 1.0, 0.0]).reshape(&[2, 2]);

    // m^0 should be the identity matrix
    let m0 = matrix_power(&m, 0).unwrap();
    assert_relative_eq!(m0.get(&[0, 0]).unwrap(), 1.0, epsilon = TOLERANCE);
    assert_relative_eq!(m0.get(&[0, 1]).unwrap(), 0.0, epsilon = TOLERANCE);
    assert_relative_eq!(m0.get(&[1, 0]).unwrap(), 0.0, epsilon = TOLERANCE);
    assert_relative_eq!(m0.get(&[1, 1]).unwrap(), 1.0, epsilon = TOLERANCE);

    // m^1 should be m
    let m1 = matrix_power(&m, 1).unwrap();
    assert_relative_eq!(m1.get(&[0, 0]).unwrap(), 1.0, epsilon = TOLERANCE);
    assert_relative_eq!(m1.get(&[0, 1]).unwrap(), 1.0, epsilon = TOLERANCE);
    assert_relative_eq!(m1.get(&[1, 0]).unwrap(), 1.0, epsilon = TOLERANCE);
    assert_relative_eq!(m1.get(&[1, 1]).unwrap(), 0.0, epsilon = TOLERANCE);

    // m^2 should be m*m = [2 1; 1 1]
    let m2 = matrix_power(&m, 2).unwrap();
    assert_relative_eq!(m2.get(&[0, 0]).unwrap(), 2.0, epsilon = TOLERANCE);
    assert_relative_eq!(m2.get(&[0, 1]).unwrap(), 1.0, epsilon = TOLERANCE);
    assert_relative_eq!(m2.get(&[1, 0]).unwrap(), 1.0, epsilon = TOLERANCE);
    assert_relative_eq!(m2.get(&[1, 1]).unwrap(), 1.0, epsilon = TOLERANCE);

    // m^5 should be [8 5; 5 3]
    let m5 = matrix_power(&m, 5).unwrap();
    assert_relative_eq!(m5.get(&[0, 0]).unwrap(), 8.0, epsilon = TOLERANCE);
    assert_relative_eq!(m5.get(&[0, 1]).unwrap(), 5.0, epsilon = TOLERANCE);
    assert_relative_eq!(m5.get(&[1, 0]).unwrap(), 5.0, epsilon = TOLERANCE);
    assert_relative_eq!(m5.get(&[1, 1]).unwrap(), 3.0, epsilon = TOLERANCE);

    // m^-2 should be (m^-1)^2 = [1 -1; -1 2] -- np.linalg.matrix_power(m,
    // -2) (numpy 2.4.2). `n = -2` is the case that specifically exercises
    // `binary_pow` on the negative side (`n = -1` above is handled by its
    // own special case in `matrix_power`, never reaching `binary_pow` at
    // all).
    let m_neg2 = matrix_power(&m, -2).unwrap();
    assert_relative_eq!(m_neg2.get(&[0, 0]).unwrap(), 1.0, epsilon = TOLERANCE);
    assert_relative_eq!(m_neg2.get(&[0, 1]).unwrap(), -1.0, epsilon = TOLERANCE);
    assert_relative_eq!(m_neg2.get(&[1, 0]).unwrap(), -1.0, epsilon = TOLERANCE);
    assert_relative_eq!(m_neg2.get(&[1, 1]).unwrap(), 2.0, epsilon = TOLERANCE);

    // m^-3 should be [-1 2; 2 -3] -- np.linalg.matrix_power(m, -3), and an
    // odd negative exponent, exercising `binary_pow`'s "extra multiply on
    // an odd bit" branch on the inverted base.
    let m_neg3 = matrix_power(&m, -3).unwrap();
    assert_relative_eq!(m_neg3.get(&[0, 0]).unwrap(), -1.0, epsilon = TOLERANCE);
    assert_relative_eq!(m_neg3.get(&[0, 1]).unwrap(), 2.0, epsilon = TOLERANCE);
    assert_relative_eq!(m_neg3.get(&[1, 0]).unwrap(), 2.0, epsilon = TOLERANCE);
    assert_relative_eq!(m_neg3.get(&[1, 1]).unwrap(), -3.0, epsilon = TOLERANCE);
}

/*
#[cfg(feature = "matrix_decomp")]
#[test]
fn test_pinv_reference() {
    // Test pseudoinverse against known values
    // Note: pinv function is not currently available in the module
    // This test is commented out until pinv is implemented

    // Invertible square matrix
    let square = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let pinv_square = pinv(&square, None).unwrap();

    // Expected pseudoinverse equals inverse for invertible matrix
    // [-2.0  1.0]
    // [ 1.5 -0.5]
    assert_relative_eq!(pinv_square.get(&[0, 0]).unwrap(), -2.0, epsilon = TOLERANCE);
    assert_relative_eq!(pinv_square.get(&[0, 1]).unwrap(), 1.0, epsilon = TOLERANCE);
    assert_relative_eq!(pinv_square.get(&[1, 0]).unwrap(), 1.5, epsilon = TOLERANCE);
    assert_relative_eq!(pinv_square.get(&[1, 1]).unwrap(), -0.5, epsilon = TOLERANCE);

    // Non-square matrix
    let rect = create_rectangle_matrix();
    let pinv_rect = pinv(&rect, None).unwrap();

    // Check pinv_rect * rect is approximately identity
    let product = pinv_rect.matmul(&rect).unwrap();

    // Should be 3x3 identity
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(product.get(&[i, j]).unwrap(), expected, epsilon = 0.001);
        }
    }
}
*/

#[cfg(feature = "matrix_decomp")]
#[test]
fn test_schur_decomposition_reference() {
    // Test Schur decomposition

    // Create a test matrix
    let m =
        Array::<f64>::from_vec(vec![3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 3.0]).reshape(&[3, 3]);

    // Compute Schur decomposition: A = Q * T * Q^T
    #[allow(deprecated)]
    let (q, t) = schur(&m).unwrap();

    // Check Q is orthogonal
    let q_t = q.transpose();
    let q_q_t = q.matmul(&q_t).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(q_q_t.get(&[i, j]).unwrap(), expected, epsilon = TOLERANCE);
        }
    }

    // Check T is upper triangular (or quasi-triangular for real Schur)
    for i in 0..3 {
        for j in 0..3 {
            if i > j + 1 {
                // Allow for 2x2 blocks in real Schur
                assert_abs_diff_eq!(t.get(&[i, j]).unwrap(), 0.0, epsilon = TOLERANCE);
            }
        }
    }

    // Check A = Q * T * Q^T
    let q_t_q_t = q.matmul(&t).unwrap().matmul(&q_t).unwrap();

    // Check reconstruction: A = Q·T·Qᵀ
    let mut reconstruction_error = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let diff = (q_t_q_t.get(&[i, j]).unwrap() - m.get(&[i, j]).unwrap()).abs();
            if diff > reconstruction_error {
                reconstruction_error = diff;
            }
        }
    }
    assert!(
        reconstruction_error < TOLERANCE,
        "Schur reconstruction error: {}",
        reconstruction_error
    );

    // Check that Q and T are the right shapes
    assert_eq!(q.shape(), &[3, 3]);
    assert_eq!(t.shape(), &[3, 3]);
}

#[test]
fn test_inner_outer_product_reference() {
    // Test inner and outer products against known values

    // Create test vectors
    let a = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array::<f64>::from_vec(vec![4.0, 5.0, 6.0]);

    // Inner product should be 1*4 + 2*5 + 3*6 = 32
    let inner_ab = inner(&a, &b).unwrap();
    assert_relative_eq!(inner_ab, 32.0, epsilon = TOLERANCE);

    // Outer product should be
    // [ 1*4 1*5 1*6 ]   [ 4  5  6 ]
    // [ 2*4 2*5 2*6 ] = [ 8 10 12 ]
    // [ 3*4 3*5 3*6 ]   [12 15 18 ]
    let outer_ab = outer(&a, &b).unwrap();

    let expected_outer = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0];
    let outer_ab_vec = outer_ab.to_vec();

    for (actual, expected) in outer_ab_vec.iter().zip(expected_outer.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = TOLERANCE);
    }
}

#[test]
fn test_vdot_reference() {
    use numrs2::linalg::vector_ops::{
        complex_vdot, vdot, ComplexVectorDotProduct, RealVectorDotProduct,
    };
    use scirs2_core::Complex;

    // Test real vdot (function-based)
    let a_real = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let b_real = Array::from_vec(vec![4.0, 5.0, 6.0]);
    let result_real = vdot(&a_real, &b_real).unwrap();
    assert_abs_diff_eq!(result_real, 32.0, epsilon = 1e-10);

    // Test real vdot (trait-based)
    let result_real_trait = a_real.vdot(&b_real).unwrap();
    assert_abs_diff_eq!(result_real_trait, 32.0, epsilon = 1e-10);

    // Test complex vdot (function-based)
    let a_complex = Array::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    let b_complex = Array::from_vec(vec![Complex::new(5.0, 6.0), Complex::new(7.0, 8.0)]);

    // vdot for complex arrays conjugates the first argument
    // So we expect: conj(1+2i) * (5+6i) + conj(3+4i) * (7+8i)
    //              = (1-2i) * (5+6i) + (3-4i) * (7+8i)
    //              = (5+6i-10i+12) + (21+24i-28i+32)
    //              = (17-4i) + (53-4i) = 70-8i
    let result_complex = complex_vdot(&a_complex, &b_complex).unwrap();
    assert_abs_diff_eq!(result_complex.re, 70.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result_complex.im, -8.0, epsilon = 1e-10);

    // Test complex vdot (trait-based)
    let result_complex_trait = a_complex.vdot(&b_complex).unwrap();
    assert_abs_diff_eq!(result_complex_trait.re, 70.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result_complex_trait.im, -8.0, epsilon = 1e-10);
}

#[test]
fn test_tensordot_reference() {
    // Test tensor contraction against known values

    // Create 3D tensors
    let a = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3, 1]);

    let b = Array::<f64>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&[3, 2, 1]);

    // Contract on axes 1 and 0 (second axis of a, first axis of b)
    // Only passing the first axis array
    let _axes_a = &[1];

    // Skip test since the API has changed
    println!("Note: tensordot API has changed and now requires different parameters");

    // Create a simpler test using 2D matrix multiplication instead
    let a_2d = a.reshape(&[2, 3]);
    let b_2d = b.reshape(&[3, 2]);
    let c = a_2d.matmul(&b_2d).unwrap().reshape(&[2, 2, 1, 1]);

    // Expected shape is [2, 2, 1, 1]
    assert_eq!(c.shape(), vec![2, 2, 1, 1]);

    // Expected values
    // c[0,0,0,0] = a[0,0,0]*b[0,0,0] + a[0,1,0]*b[1,0,0] + a[0,2,0]*b[2,0,0]
    //            = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // c[0,1,0,0] = a[0,0,0]*b[0,1,0] + a[0,1,0]*b[1,1,0] + a[0,2,0]*b[2,1,0]
    //            = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // c[1,0,0,0] = a[1,0,0]*b[0,0,0] + a[1,1,0]*b[1,0,0] + a[1,2,0]*b[2,0,0]
    //            = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // c[1,1,0,0] = a[1,0,0]*b[0,1,0] + a[1,1,0]*b[1,1,0] + a[1,2,0]*b[2,1,0]
    //            = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

    assert_relative_eq!(c.get(&[0, 0, 0, 0]).unwrap(), 58.0, epsilon = TOLERANCE);
    assert_relative_eq!(c.get(&[0, 1, 0, 0]).unwrap(), 64.0, epsilon = TOLERANCE);
    assert_relative_eq!(c.get(&[1, 0, 0, 0]).unwrap(), 139.0, epsilon = TOLERANCE);
    assert_relative_eq!(c.get(&[1, 1, 0, 0]).unwrap(), 154.0, epsilon = TOLERANCE);
}

/// Regression test for the previously-`NotImplemented` general-axes hole in
/// `tensordot` (`src/linalg/tensor_ops.rs`, the `a_axis == 0 && b_axis == 1`
/// case used to silently return a wrongly-transposed result instead, and
/// every other axis combination on non-2-D inputs errored outright).
///
/// Reference: `np.tensordot(A, B, axes=([0], [1]))` (numpy 2.4.2) with
/// `A = [[1,2,3],[4,5,6]]` (2x3), `B = [[1,2],[3,4],[5,6],[7,8]]` (4x2) ->
/// `[[9,19,29,39],[12,26,40,54],[15,33,51,69]]`.
#[test]
fn test_tensordot_axis0_axis1_reference() {
    let a = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let b = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(&[4, 2]);

    let result = tensordot(&a, &b, &[0, 1]).expect("tensordot should succeed");
    assert_eq!(result.shape(), vec![3, 4]);

    #[rustfmt::skip]
    let expected = [
        [9.0, 19.0, 29.0, 39.0],
        [12.0, 26.0, 40.0, 54.0],
        [15.0, 33.0, 51.0, 69.0],
    ];
    for i in 0..3 {
        for j in 0..4 {
            assert_relative_eq!(
                result.get(&[i, j]).unwrap(),
                expected[i][j],
                epsilon = TOLERANCE
            );
        }
    }
}

/// General N-D `tensordot`, contracting a non-edge axis of each 3-D operand
/// -- the case the old implementation rejected outright (it only ever
/// handled 2-D operands). Reference: `np.tensordot(a, b, axes=([1], [1]))`
/// on `a = np.random.RandomState(42).rand(2, 3, 4)`, `b =
/// np.random.RandomState(42)`-continued `.rand(5, 3, 6)` (numpy 2.4.2).
#[test]
fn test_tensordot_general_3d_reference() {
    #[rustfmt::skip]
    let a = Array::<f64>::from_vec(vec![
        0.3745401188473625, 0.9507143064099162, 0.7319939418114051, 0.5986584841970366,
        0.15601864044243652, 0.15599452033620265, 0.05808361216819946, 0.8661761457749352,
        0.6011150117432088, 0.7080725777960455, 0.02058449429580245, 0.9699098521619943,
        0.8324426408004217, 0.21233911067827616, 0.18182496720710062, 0.18340450985343382,
        0.3042422429595377, 0.5247564316322378, 0.43194501864211576, 0.2912291401980419,
        0.6118528947223795, 0.13949386065204183, 0.29214464853521815, 0.3663618432936917,
    ])
    .reshape(&[2, 3, 4]);

    #[rustfmt::skip]
    let b = Array::<f64>::from_vec(vec![
        0.45606998421703593, 0.7851759613930136, 0.19967378215835974, 0.5142344384136116, 0.5924145688620425, 0.04645041271999772,
        0.6075448519014384, 0.17052412368729153, 0.06505159298527952, 0.9488855372533332, 0.9656320330745594, 0.8083973481164611,
        0.3046137691733707, 0.09767211400638387, 0.6842330265121569, 0.4401524937396013, 0.12203823484477883, 0.4951769101112702,
        0.0343885211152184, 0.9093204020787821, 0.2587799816000169, 0.662522284353982, 0.31171107608941095, 0.5200680211778108,
        0.5467102793432796, 0.18485445552552704, 0.9695846277645586, 0.7751328233611146, 0.9394989415641891, 0.8948273504276488,
        0.5978999788110851, 0.9218742350231168, 0.0884925020519195, 0.1959828624191452, 0.04522728891053807, 0.32533033076326434,
        0.388677289689482, 0.2713490317738959, 0.8287375091519293, 0.3567533266935893, 0.28093450968738076, 0.5426960831582485,
        0.14092422497476265, 0.8021969807540397, 0.07455064367977082, 0.9868869366005173, 0.7722447692966574, 0.1987156815341724,
        0.0055221171236024, 0.8154614284548342, 0.7068573438476171, 0.7290071680409873, 0.7712703466859457, 0.07404465173409036,
        0.3584657285442726, 0.11586905952512971, 0.8631034258755935, 0.6232981268275579, 0.3308980248526492, 0.06355835028602363,
        0.3109823217156622, 0.32518332202674705, 0.7296061783380641, 0.6375574713552131, 0.8872127425763265, 0.4722149251619493,
        0.1195942459383017, 0.713244787222995, 0.7607850486168974, 0.5612771975694962, 0.770967179954561, 0.49379559636439074,
        0.5227328293819941, 0.42754101835854963, 0.02541912674409519, 0.10789142699330445, 0.03142918568673425, 0.6364104112637804,
        0.3143559810763267, 0.5085706911647028, 0.907566473926093, 0.24929222914887494, 0.41038292303562973, 0.7555511385430487,
        0.22879816549162246, 0.07697990982879299, 0.289751452913768, 0.16122128725400442, 0.9296976523425731, 0.808120379564417,
    ])
    .reshape(&[5, 3, 6]);

    let result = tensordot(&a, &b, &[1, 1]).expect("tensordot should succeed");
    // np.tensordot(a, b, axes=([1],[1])).shape == (2, 4, 5, 6)
    assert_eq!(result.shape(), vec![2, 4, 5, 6]);
    assert_relative_eq!(
        result.get(&[0, 0, 0, 0]).unwrap(),
        0.44871273732662104,
        epsilon = 1e-9
    );
    assert_relative_eq!(
        result.get(&[1, 2, 4, 5]).unwrap(),
        0.6781598970433369,
        epsilon = 1e-9
    );
}

#[test]
fn test_kron_reference() {
    // Test Kronecker product against known values

    // Create test matrices
    let a = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Array::<f64>::from_vec(vec![0.1, 0.2, 0.3, 0.4]).reshape(&[2, 2]);

    // Compute Kronecker product
    let k = kron(&a, &b).unwrap();

    // Expected shape is [4, 4]
    assert_eq!(k.shape(), vec![4, 4]);

    // Expected values
    // [ 1*0.1 1*0.2 2*0.1 2*0.2 ]   [ 0.1 0.2 0.2 0.4 ]
    // [ 1*0.3 1*0.4 2*0.3 2*0.4 ] = [ 0.3 0.4 0.6 0.8 ]
    // [ 3*0.1 3*0.2 4*0.1 4*0.2 ]   [ 0.3 0.6 0.4 0.8 ]
    // [ 3*0.3 3*0.4 4*0.3 4*0.4 ]   [ 0.9 1.2 1.2 1.6 ]

    // Define expected values row by row
    let expected = [
        0.1, 0.2, 0.2, 0.4, 0.3, 0.4, 0.6, 0.8, 0.3, 0.6, 0.4, 0.8, 0.9, 1.2, 1.2, 1.6,
    ];

    // Check all values
    let k_vec = k.to_vec();
    for (actual, expected) in k_vec.iter().zip(expected.iter()) {
        assert_relative_eq!(*actual, *expected, epsilon = TOLERANCE);
    }
}

/// Reference values from `numpy.linalg.multi_dot` (numpy 2.4.2):
/// `A1` is 3x2, `A2` is 2x3, `A3` is 3x2 -- deliberately non-square and
/// non-uniform so the matrix-chain-order DP has more than one
/// parenthesization to choose between (`(A1 @ A2) @ A3` vs.
/// `A1 @ (A2 @ A3)`), not just the single-choice 2- or 3-matrix square case.
#[test]
fn test_multi_dot_three_matrices_reference() {
    use numrs2::linalg::multi_dot;

    let a1 = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[3, 2]);
    let a2 = Array::<f64>::from_vec(vec![1.0, 0.0, 2.0, 0.0, 1.0, 1.0]).reshape(&[2, 3]);
    let a3 = Array::<f64>::from_vec(vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]).reshape(&[3, 2]);

    let result = multi_dot(&[&a1, &a2, &a3]).expect("multi_dot should succeed");
    assert_eq!(result.shape(), vec![3, 2]);

    // np.linalg.multi_dot([A1, A2, A3]) == [[5, 3], [13, 7], [21, 11]]
    let expected = [[5.0, 3.0], [13.0, 7.0], [21.0, 11.0]];
    for i in 0..3 {
        for j in 0..2 {
            assert_relative_eq!(
                result.get(&[i, j]).unwrap(),
                expected[i][j],
                epsilon = TOLERANCE
            );
        }
    }

    // Must also agree with naive sequential evaluation, independent of
    // which parenthesization the DP picked -- matrix multiplication is
    // associative, so both orders are exactly the same computation up to
    // floating-point rounding at this scale.
    let naive = a1.matmul(&a2).unwrap().matmul(&a3).unwrap();
    for i in 0..3 {
        for j in 0..2 {
            assert_relative_eq!(
                result.get(&[i, j]).unwrap(),
                naive.get(&[i, j]).unwrap(),
                epsilon = TOLERANCE
            );
        }
    }
}

/// The 2-matrix fast path (no DP needed) and vector-first/vector-last
/// squeeze handling, both required by NumPy's `multi_dot` semantics.
#[test]
fn test_multi_dot_two_matrices_and_vector_ends_reference() {
    use numrs2::linalg::multi_dot;

    let a = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Array::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    // Two matrices: must exactly match `matmul`.
    let two_result = multi_dot(&[&a, &b]).expect("2-matrix multi_dot should succeed");
    let expected_two = a.matmul(&b).unwrap();
    assert_eq!(two_result.shape(), expected_two.shape());
    for i in 0..2 {
        for j in 0..2 {
            assert_relative_eq!(
                two_result.get(&[i, j]).unwrap(),
                expected_two.get(&[i, j]).unwrap(),
                epsilon = TOLERANCE
            );
        }
    }

    // Vector-first: np.linalg.multi_dot([[1,1], A, B]) == [62, 72]
    // (v @ A == [1+3, 2+4] == [4, 6]; then [4, 6] @ B == [4*5+6*7, 4*6+6*8]
    // == [62, 72] -- confirmed against `np.linalg.multi_dot` directly).
    let v = Array::<f64>::from_vec(vec![1.0, 1.0]);
    let vec_first = multi_dot(&[&v, &a, &b]).expect("vector-first multi_dot should succeed");
    assert_eq!(vec_first.shape(), vec![2]);
    assert_relative_eq!(vec_first.get(&[0]).unwrap(), 62.0, epsilon = TOLERANCE);
    assert_relative_eq!(vec_first.get(&[1]).unwrap(), 72.0, epsilon = TOLERANCE);

    // Vector-last: np.linalg.multi_dot([A, B, [1,1]]) == [41, 93]
    // (B @ w == [5+6, 7+8] == [11, 15]; then A @ [11, 15] ==
    // [1*11+2*15, 3*11+4*15] == [41, 93]).
    let w = Array::<f64>::from_vec(vec![1.0, 1.0]);
    let vec_last = multi_dot(&[&a, &b, &w]).expect("vector-last multi_dot should succeed");
    assert_eq!(vec_last.shape(), vec![2]);
    assert_relative_eq!(vec_last.get(&[0]).unwrap(), 41.0, epsilon = TOLERANCE);
    assert_relative_eq!(vec_last.get(&[1]).unwrap(), 93.0, epsilon = TOLERANCE);
}

/// `numpy.linalg.tensorsolve`, pinned against NumPy's own doc example (with
/// a fixed seed via `numpy.random.default_rng(0)`).
#[test]
fn test_tensorsolve_reference() {
    use numrs2::linalg::tensorsolve;

    // a = eye(24).reshape(6, 4, 2, 3, 4): a "square" tensor coefficient
    // (prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim]) == 24) equal to the
    // identity in flattened form, so tensorsolve(a, b) == b reshaped to
    // a.shape()[b.ndim..] -- i.e. this reduces to a reshape, independent of
    // the actual solve path (a useful property check on top of the direct
    // value pin below).
    let a = Array::<f64>::eye(24, 24, 0).reshape(&[6, 4, 2, 3, 4]);

    // b, pinned from `numpy.random.default_rng(0).normal(size=(6, 4))`.
    #[rustfmt::skip]
    let b_data = vec![
        0.1257302210933933, -0.1321048632913019, 0.6404226504432821, 0.10490011715303971,
        -0.535669373161111, 0.36159505490948474, 1.3040000451301372, 0.9470809631292422,
        -0.7037352358069926, -1.2654214710460525, -0.6232744625373522, 0.0413259793472436,
        -2.3250307746388343, -0.21879166393254573, -1.2459109472530652, -0.7322673547034516,
        -0.5442589828573099, -0.31630015636915454, 0.4116305363741328, 1.0425133694426776,
        -0.12853466294403426, 1.3664634705496859, -0.6651946734866135, 0.3515100700930197,
    ];
    let b = Array::from_vec(b_data.clone()).reshape(&[6, 4]);

    let x = tensorsolve(&a, &b, None).expect("tensorsolve should succeed");
    assert_eq!(x.shape(), vec![2, 3, 4]);

    // Since `a` is the identity (reshaped), `x` must equal `b` reshaped to
    // `a.shape()[b.ndim..]` -- exactly `b_data` in the same flat order.
    let x_vec = x.to_vec();
    for i in 0..24 {
        assert_relative_eq!(x_vec[i], b_data[i], epsilon = 1e-9);
    }
}

/// `numpy.linalg.tensorsolve` with an explicit `axes` argument -- the
/// reorder-before-solve path (`allaxes.remove(k); allaxes.insert(an, k)`
/// per NumPy's own source), not exercised by the identity-shaped `a` above
/// (whose default `axes=None` path never permutes anything).
///
/// `a` is 4-D `(2, 2, 3, 3)`, NOT already "square" in tensorsolve's sense
/// with `b`'s 2-D `(2, 3)` prefix -- axes `(0, 3)` must be moved to the end
/// first (yielding effective shape `(2, 3, 2, 3)`) before it becomes
/// solvable. Reference values from `numpy.linalg.tensorsolve(a, b,
/// axes=(0, 3))` (numpy 2.4.2, `a`/`b` from
/// `numpy.random.default_rng(5).normal(...)`).
#[test]
fn test_tensorsolve_with_axes_reference() {
    use numrs2::linalg::tensorsolve;

    #[rustfmt::skip]
    let a_data = vec![
        -0.8019314252534474, -1.324358995628145, -0.24836162209524854,
        0.4204452380655215, 1.1360465324896427, 0.10970639932180819,
        -0.5526473205362324, -0.7847803553442784, 0.7487457707345911,
        1.6347830429585775, 0.27276877584472176, -1.2333286640307717,
        -0.9582652054360887, 1.6000190889991115, 0.2028824405086084,
        -1.7321348424395848, -0.08369619281702581, -1.1632259734447485,
        -0.6292880940615545, -0.48800582327685743, -0.7133133716322436,
        0.5533784703532895, -0.06308597192528916, -0.5894312580326048,
        0.40963782655711695, 0.8298553070613239, -1.643023371405677,
        -0.256730126365494, -0.9807473560440125, -0.17315522486203205,
        -1.2894187467538587, 0.0206903940375912, -0.03788574104406823,
        -0.304337750958489, -1.0479265051202462, -0.3961903304730927,
    ];
    let a = Array::from_vec(a_data).reshape(&[2, 2, 3, 3]);

    #[rustfmt::skip]
    let b_data = vec![
        -1.091328901695709, -1.3552087462047395, 0.22478573245989314,
        -1.109349937891366, 1.1702961011782933, 0.7165876558738361,
    ];
    let b = Array::from_vec(b_data).reshape(&[2, 3]);

    let x = tensorsolve(&a, &b, Some(&[0, 3])).expect("tensorsolve with axes should succeed");
    assert_eq!(x.shape(), vec![2, 3]);

    #[rustfmt::skip]
    let expected = [
        -0.7663260286978177, 0.2505573754360905, -3.7309172928094645,
        -0.5546132418172874, 4.668466795395245, 0.5207123902231279,
    ];
    let x_vec = x.to_vec();
    for i in 0..6 {
        assert_relative_eq!(x_vec[i], expected[i], epsilon = 1e-8);
    }
}

/// A duplicate entry in `tensorsolve`'s `axes` must be rejected with a
/// normal `Err`, not panic. This is not tested by NumPy conformance (NumPy
/// itself does not validate `axes` for duplicates either, and silently
/// produces a nonsensical permutation instead) -- it targets this crate's
/// own defensive check, added because a duplicate can otherwise underflow
/// `an - k` (`k = axes.len()` counting the duplicate twice, so it can
/// exceed `an` even though every individual entry is in-bounds) before
/// `moveaxis` is ever reached.
#[test]
fn test_tensorsolve_duplicate_axes_errors() {
    use numrs2::linalg::tensorsolve;

    let a = Array::<f64>::eye(4, 4, 0).reshape(&[2, 2, 2, 2]);
    let b = Array::<f64>::from_vec(vec![1.0, 2.0]);

    let result = tensorsolve(&a, &b, Some(&[0, 0, 0]));
    assert!(
        result.is_err(),
        "duplicate axes entries must error, not panic or silently misbehave"
    );
}

/// `numpy.linalg.tensorinv`, pinned against NumPy's own doc example: `a =
/// eye(24).reshape(24, 8, 3)`, `ind=1`. Also exercises the identity this
/// operation is defined by (`tensordot(tensorinv(a), b, axes=1) ==
/// tensorsolve(a, b)`) against a fixed-seed random `b` -- `ind=1` and a 1-D
/// `b` keep the verification `tensordot` call to a single contracted axis
/// pair, matching this crate's `tensordot(&a, &b, &[a_axis, b_axis])`
/// signature exactly (unlike NumPy's `axes=N` form, which can contract
/// several trailing/leading axes at once).
#[test]
fn test_tensorinv_reference() {
    use numrs2::linalg::{tensordot, tensorinv, tensorsolve};

    let a = Array::<f64>::eye(24, 24, 0).reshape(&[24, 8, 3]);
    let a_inv = tensorinv(&a, 1).expect("tensorinv should succeed");
    assert_eq!(a_inv.shape(), vec![8, 3, 24]);

    // b, pinned from `numpy.random.default_rng(2).normal(size=24)`.
    #[rustfmt::skip]
    let b_data = vec![
        0.18905338179353307, -0.5227484414807474, -0.41306354339189344,
        -2.4414673826398556, 1.799707382720902, 1.1441658720372287,
        -0.32542283686782436, 0.7738065867276614, 0.28121066979764925,
        -0.5538228364240524, 0.9775674511260357, -0.31055654665915255,
        -0.3288239040579627, -0.7921467553588982, 0.45495807124085547,
        -0.09919805171738795, 0.5452887139646817, -0.6071856998706371,
        0.12682784711186987, -0.8922740434297903, 0.8414649723701431,
        0.18803508698068597, 0.33057100813532614, 0.41050391297026284,
    ];
    let b = Array::from_vec(b_data);

    // `tensordot(ainv, b, axes=1)`: contract ainv's last axis (index 2)
    // against b's only axis (index 0) -- a single axis pair.
    let lhs = tensordot(&a_inv, &b, &[2, 0]).expect("tensordot should succeed");
    let rhs = tensorsolve(&a, &b, None).expect("tensorsolve should succeed");

    assert_eq!(lhs.shape(), vec![8, 3]);
    assert_eq!(rhs.shape(), vec![8, 3]);

    let lhs_vec = lhs.to_vec();
    let rhs_vec = rhs.to_vec();
    for i in 0..lhs_vec.len() {
        assert_relative_eq!(lhs_vec[i], rhs_vec[i], epsilon = 1e-9);
    }

    // Reference values: `np.tensordot(np.linalg.tensorinv(a, ind=1), b,
    // axes=1)` with the `a`/`b` above.
    #[rustfmt::skip]
    let expected = [
        [0.18905338179353307, -0.5227484414807474, -0.41306354339189344],
        [-2.4414673826398556, 1.799707382720902, 1.1441658720372287],
        [-0.32542283686782436, 0.7738065867276614, 0.28121066979764925],
        [-0.5538228364240524, 0.9775674511260357, -0.31055654665915255],
        [-0.3288239040579627, -0.7921467553588982, 0.45495807124085547],
        [-0.09919805171738795, 0.5452887139646817, -0.6071856998706371],
        [0.12682784711186987, -0.8922740434297903, 0.8414649723701431],
        [0.18803508698068597, 0.33057100813532614, 0.41050391297026284],
    ];
    for i in 0..8 {
        for j in 0..3 {
            assert_relative_eq!(lhs.get(&[i, j]).unwrap(), expected[i][j], epsilon = 1e-9);
        }
    }
}
