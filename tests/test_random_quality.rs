//! Regression tests for W1-F: random distribution correctness.
//!
//! Covers:
//! 1. Sobol sequence: proper Joe-Kuo direction numbers (dims 1..=40),
//!    cross-checked against known `scipy.stats.qmc.Sobol` values and the
//!    stratification (equidistribution) property for dimensions beyond the
//!    old 8-dimension hardcoded table.
//! 2. `nearest_correlation_matrix`: Higham's alternating-projections
//!    algorithm, cross-checked against the published example from
//!    Higham (2002), "Computing the nearest correlation matrix -- a problem
//!    from finance".
//!
//! (erf and student_t_cdf precision regressions live in the in-file
//! `mod tests` of `src/random/distributions_enhanced.rs`, since they target
//! private helper functions that this external test crate cannot see.)

use numrs2::array::Array;
use numrs2::linalg_stable::StableDecompositions;
use numrs2::random::distributions_enhanced::nearest_correlation_matrix;
use numrs2::random::random_correlation_matrix;
use numrs2::random::sobol_sequence;
use serial_test::serial;

// ---------------------------------------------------------------------
// Sobol sequence
// ---------------------------------------------------------------------

/// dim=1 is the trivial base-2 (van der Corput-like) Sobol sequence. Values
/// verified bit-for-bit against `scipy.stats.qmc.Sobol(d=1, scramble=False)`.
#[test]
fn test_sobol_dim1_matches_scipy_known_values() {
    let samples = sobol_sequence::<f64>(1, 8).expect("test: sobol_sequence should succeed");
    let data = samples.to_vec();

    let expected = [0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125];
    assert_eq!(data.len(), expected.len());
    for (i, (&got, &want)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "point {i}: got {got}, expected {want}"
        );
    }
}

/// dim=2 first 8 points, verified bit-for-bit against
/// `scipy.stats.qmc.Sobol(d=2, scramble=False).random_base2(m=3)`.
#[test]
fn test_sobol_dim2_matches_scipy_known_values() {
    let samples = sobol_sequence::<f64>(2, 8).expect("test: sobol_sequence should succeed");
    let data = samples.to_vec();

    #[rustfmt::skip]
    let expected: [f64; 16] = [
        0.0,   0.0,
        0.5,   0.5,
        0.75,  0.25,
        0.25,  0.75,
        0.375, 0.375,
        0.875, 0.875,
        0.625, 0.125,
        0.125, 0.625,
    ];
    assert_eq!(data.len(), expected.len());
    for (i, (&got, &want)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "entry {i}: got {got}, expected {want}"
        );
    }
}

/// Property test: for dimensions beyond the old hardcoded 8-dimension
/// table (d in {1, 5, 9, 17, 33}), the first 2^k points' d-th coordinate
/// must hit each of the 2^k equal-width bins exactly once (the defining
/// stratification / low-discrepancy property of a Sobol sequence). Before
/// the Joe-Kuo table was completed, dimensions above 8 fell back to
/// pseudorandom values and failed this property.
#[test]
fn test_sobol_stratification_beyond_old_8dim_table() {
    let k = 4usize;
    let n = 1usize << k; // 16 points

    for &dim in &[1usize, 5, 9, 17, 33] {
        let samples = sobol_sequence::<f64>(dim, n).unwrap_or_else(|e| {
            panic!("test: sobol_sequence(dim={dim}, n={n}) should succeed: {e:?}")
        });
        let data = samples.to_vec();

        // Check every coordinate (0..dim), not just the last one, since a
        // pseudorandom fallback for some coordinates but not others would
        // otherwise slip through.
        for d in 0..dim {
            let mut bin_hits = vec![0usize; n];
            for i in 0..n {
                let v = data[i * dim + d];
                assert!(
                    (0.0..1.0).contains(&v),
                    "dim={dim} coord={d} point={i}: value {v} out of [0,1)"
                );
                let mut bin = (v * n as f64).floor() as usize;
                if bin >= n {
                    bin = n - 1;
                }
                bin_hits[bin] += 1;
            }
            for (bin, &hits) in bin_hits.iter().enumerate() {
                assert_eq!(
                    hits, 1,
                    "dim={dim} coord={d}: bin {bin} hit {hits} times (expected exactly 1) -- \
                     stratification violated, points: {data:?}"
                );
            }
        }
    }
}

#[test]
fn test_sobol_dim_zero_is_rejected() {
    let result = sobol_sequence::<f64>(0, 4);
    assert!(result.is_err(), "dim=0 should be rejected");
}

#[test]
fn test_sobol_dim_above_40_is_rejected() {
    let result = sobol_sequence::<f64>(41, 4);
    assert!(
        result.is_err(),
        "dim=41 should be rejected (embedded Joe-Kuo table covers dims 1..=40)"
    );
}

#[test]
fn test_sobol_dim_40_succeeds() {
    let result = sobol_sequence::<f64>(40, 4);
    assert!(result.is_ok(), "dim=40 (table boundary) should succeed");
}

#[test]
fn test_sobol_points_in_unit_interval() {
    // All coordinates of all generated points must lie in [0, 1).
    for &dim in &[1usize, 8, 20, 40] {
        let samples = sobol_sequence::<f64>(dim, 64).expect("test: sobol_sequence should succeed");
        for v in samples.to_vec() {
            assert!((0.0..1.0).contains(&v), "dim={dim}: value {v} out of [0,1)");
        }
    }
}

// ---------------------------------------------------------------------
// nearest_correlation_matrix (Higham's alternating projections)
// ---------------------------------------------------------------------

/// The classic non-PSD example from Higham (2002), "Computing the nearest
/// correlation matrix -- a problem from finance", Section 1. The matrix has
/// eigenvalues (-0.4142, 1, 2.4142), i.e. it is indefinite. The paper's
/// nearest correlation matrix has off-diagonal entries 0.7607 and 0.1573
/// (rounded to 4 places); this is independently reproduced by a validated
/// numpy prototype of the same algorithm (off-diagonals 0.76068985,
/// 0.15729811 before rounding).
#[test]
fn test_nearest_correlation_matrix_higham_example() {
    #[rustfmt::skip]
    let a: Array<f64> = Array::from_vec(vec![
        1.0, 1.0, 0.0,
        1.0, 1.0, 1.0,
        0.0, 1.0, 1.0,
    ])
    .reshape(&[3, 3]);

    let result = nearest_correlation_matrix(&a)
        .expect("test: nearest_correlation_matrix should converge on the Higham example");
    let data = result.to_vec();

    // Unit diagonal.
    for i in 0..3 {
        let d = data[i * 3 + i];
        assert!((d - 1.0).abs() < 1e-8, "diag[{i}] = {d}, expected 1.0");
    }

    // Symmetric.
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (data[i * 3 + j] - data[j * 3 + i]).abs() < 1e-10,
                "not symmetric at ({i},{j})"
            );
        }
    }

    // Matches the published/independently-reproduced off-diagonal values.
    assert!(
        (data[1] - 0.760_689_85).abs() < 1e-4,
        "corr[0,1] = {}, expected ~0.76069",
        data[1]
    );
    assert!(
        (data[2] - 0.157_298_11).abs() < 1e-4,
        "corr[0,2] = {}, expected ~0.15730",
        data[2]
    );

    // Minimum eigenvalue must be numerically non-negative (PSD).
    let min_eig = min_eigenvalue_3x3_symmetric(&data);
    assert!(
        min_eig >= -1e-10,
        "min eigenvalue {min_eig} should be >= -1e-10 (PSD)"
    );
}

/// An already-valid correlation matrix (symmetric, unit diagonal, PSD)
/// should be an (approximate) fixed point of the projection.
#[test]
fn test_nearest_correlation_matrix_already_valid_returns_itself() {
    #[rustfmt::skip]
    let b: Array<f64> = Array::from_vec(vec![
        1.0, 0.5, 0.3,
        0.5, 1.0, 0.2,
        0.3, 0.2, 1.0,
    ])
    .reshape(&[3, 3]);

    let result = nearest_correlation_matrix(&b)
        .expect("test: nearest_correlation_matrix should succeed on an already-valid matrix");
    let data = result.to_vec();
    let original = b.to_vec();

    for (i, (&got, &want)) in data.iter().zip(original.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-8,
            "entry {i}: got {got}, expected ~{want} (already-valid input should be near-fixed)"
        );
    }
}

/// Output must always be symmetric with a unit diagonal and be PSD, even
/// for a more strongly indefinite input than the classic 3x3 example.
#[test]
fn test_nearest_correlation_matrix_strongly_indefinite_input() {
    #[rustfmt::skip]
    let a: Array<f64> = Array::from_vec(vec![
         1.0, 0.9, -0.9, 0.9,
         0.9, 1.0,  0.9, -0.9,
        -0.9, 0.9,  1.0, 0.9,
         0.9, -0.9, 0.9, 1.0,
    ])
    .reshape(&[4, 4]);

    let result =
        nearest_correlation_matrix(&a).expect("test: nearest_correlation_matrix should converge");
    let data = result.to_vec();
    let n = 4;

    for i in 0..n {
        assert!(
            (data[i * n + i] - 1.0).abs() < 1e-8,
            "diag[{i}] = {}",
            data[i * n + i]
        );
        for j in 0..n {
            assert!(
                (data[i * n + j] - data[j * n + i]).abs() < 1e-10,
                "not symmetric at ({i},{j})"
            );
        }
    }

    let result_array = Array::from_vec(data).reshape(&[n, n]);
    let (eigenvalues, _) = StableDecompositions::symmetric_eigendecomposition(&result_array)
        .expect("test: symmetric_eigendecomposition should succeed on the projected result");
    let min_eig = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        min_eig >= -1e-8,
        "min eigenvalue {min_eig} should be (numerically) >= 0"
    );
}

#[test]
fn test_nearest_correlation_matrix_rejects_non_square() {
    let a: Array<f64> = Array::from_vec(vec![1.0, 0.5, 0.5, 1.0, 0.2, 0.3]).reshape(&[2, 3]);
    let result = nearest_correlation_matrix(&a);
    assert!(result.is_err(), "non-square input should be rejected");
}

/// `random_correlation_matrix` is the one real production caller of the new
/// Higham projection: it feeds it a random symmetric matrix with off-diagonal
/// entries uniform on [-1, 1] and a unit diagonal, which is far more
/// indefinite in general than the hand-picked examples above. This test
/// exercises that actual code path (not just `nearest_correlation_matrix`
/// directly) across a few sizes to confirm the projection reliably converges
/// within `NEAREST_CORRELATION_MAX_ITER` on realistic random inputs, and that
/// its output is a valid correlation matrix (symmetric, unit diagonal, PSD).
/// Draws from the global RNG, so this must run serially with other such
/// tests.
#[test]
#[serial]
fn test_random_correlation_matrix_converges_and_is_valid() {
    for &n in &[2usize, 3, 5, 10, 20, 30, 50] {
        let result = random_correlation_matrix::<f64>(n);
        let corr = result.unwrap_or_else(|e| {
            panic!("random_correlation_matrix(n={n}) should converge, got error: {e:?}")
        });
        let data = corr.to_vec();
        assert_eq!(data.len(), n * n);

        for i in 0..n {
            assert!(
                (data[i * n + i] - 1.0).abs() < 1e-6,
                "n={n}: diag[{i}] = {}, expected 1.0",
                data[i * n + i]
            );
            for j in 0..n {
                assert!(
                    (data[i * n + j] - data[j * n + i]).abs() < 1e-8,
                    "n={n}: not symmetric at ({i},{j})"
                );
            }
        }

        let corr_array = Array::from_vec(data).reshape(&[n, n]);
        let (eigenvalues, _) = StableDecompositions::symmetric_eigendecomposition(&corr_array)
            .unwrap_or_else(|e| {
                panic!("n={n}: symmetric_eigendecomposition on the result should succeed: {e:?}")
            });
        let min_eig = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            min_eig >= -1e-6,
            "n={n}: min eigenvalue {min_eig} should be (numerically) >= 0"
        );
    }
}

// ---------------------------------------------------------------------
// Small local helper for the 3x3 eigenvalue sanity check below. Deliberately
// independent of the crate's own eigendecomposition (closed-form, not
// iterative) so that test doesn't just check the implementation against
// itself; validated against numpy.linalg.eigvalsh over 2000 random
// symmetric 3x3 matrices (max abs error ~7e-15). The 4x4 test further down
// instead calls the crate's own `StableDecompositions::symmetric_eigendecomposition`
// directly, which is fine there: `nearest_correlation_matrix` only clips
// eigenvalues of *intermediate* iterates, so confirming the *converged,
// unit-diagonal* output is still PSD is a genuine end-to-end check, not a
// tautology.
// ---------------------------------------------------------------------

/// Closed-form minimum eigenvalue of a symmetric 3x3 matrix via the
/// trigonometric method (Smith, 1961) -- avoids depending on the crate's
/// own eigendecomposition for this correctness check.
fn min_eigenvalue_3x3_symmetric(m: &[f64]) -> f64 {
    let a = m; // row-major 3x3
    let p1 = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
    if p1 < 1e-14 {
        // Already diagonal.
        return a[0].min(a[4]).min(a[8]);
    }
    let q = (a[0] + a[4] + a[8]) / 3.0;
    let p2 = (a[0] - q).powi(2) + (a[4] - q).powi(2) + (a[8] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();

    let mut b = [0.0f64; 9];
    for i in 0..9 {
        b[i] = a[i] / p;
    }
    b[0] -= q / p;
    b[4] -= q / p;
    b[8] -= q / p;

    let det_b = b[0] * (b[4] * b[8] - b[5] * b[7]) - b[1] * (b[3] * b[8] - b[5] * b[6])
        + b[2] * (b[3] * b[7] - b[4] * b[6]);
    let r = (det_b / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    let eig1 = q + 2.0 * p * phi.cos();
    let eig3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    let eig2 = 3.0 * q - eig1 - eig3;

    eig1.min(eig2).min(eig3)
}
