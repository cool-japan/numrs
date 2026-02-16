//! Eigenvalue decomposition (Jacobi method for symmetric matrices).

use crate::error::{NumRs2Error, Result};

/// Symmetric eigendecomposition using Jacobi iteration.
///
/// This is a pure-Rust implementation of the classical Jacobi eigenvalue
/// algorithm for symmetric matrices. It is suitable for the covariance matrices
/// encountered in CMA-ES (typically moderate dimension, n < 200).
///
/// The algorithm works by applying a sequence of plane rotations (Givens
/// rotations) to zero out off-diagonal elements, converging to a diagonal
/// matrix of eigenvalues with the accumulated rotations forming the
/// eigenvector matrix.
///
/// # Arguments
/// * `matrix` - Input symmetric matrix stored as row-major flat vector (n x n)
/// * `n` - Matrix dimension
///
/// # Returns
/// `(eigenvalues, eigenvectors)` where eigenvectors are stored column-wise
/// in row-major order (i.e., the k-th eigenvector is at column k).
///
/// # Errors
/// Returns an error if the matrix size does not match the specified dimension.
pub(crate) fn symmetric_eigendecomposition(
    matrix: &[f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if matrix.len() != n * n {
        return Err(NumRs2Error::InvalidInput(format!(
            "Matrix size {} does not match dimension {}x{}",
            matrix.len(),
            n,
            n
        )));
    }

    if n == 1 {
        return Ok((vec![matrix[0]], vec![1.0]));
    }

    // Work on a copy
    let mut a = matrix.to_vec();

    // Initialize eigenvector matrix as identity
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_sweeps = 100 * n * n;
    let tolerance = 1e-15;

    for _sweep in 0..max_sweeps {
        // Compute off-diagonal Frobenius norm
        let mut off_diag_sq = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off_diag_sq += a[i * n + j] * a[i * n + j];
            }
        }
        let off_diag = (2.0 * off_diag_sq).sqrt();

        // Check convergence
        let diag_norm: f64 = (0..n)
            .map(|i| a[i * n + i] * a[i * n + i])
            .sum::<f64>()
            .sqrt();
        if off_diag < tolerance * diag_norm.max(1.0) {
            break;
        }

        // Perform Jacobi rotations over all (p, q) pairs
        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = a[p * n + q];
                if a_pq.abs() < 1e-30 {
                    continue;
                }

                let a_pp = a[p * n + p];
                let a_qq = a[q * n + q];
                let tau = (a_qq - a_pp) / (2.0 * a_pq);

                // Compute t = sign(tau) / (|tau| + sqrt(1 + tau^2))
                // This formula avoids cancellation for large |tau|
                let t = if tau.abs() > 1e15 {
                    1.0 / (2.0 * tau)
                } else {
                    let sign_tau = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign_tau / (tau.abs() + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply Givens rotation to matrix A
                apply_jacobi_rotation_matrix(&mut a, n, p, q, c, s);

                // Update eigenvectors
                apply_jacobi_rotation_vectors(&mut v, n, p, q, c, s);
            }
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();

    Ok((eigenvalues, v))
}

/// Apply a Jacobi (Givens) rotation to the symmetric matrix A.
///
/// Zeros out the (p,q) off-diagonal element while preserving symmetry.
fn apply_jacobi_rotation_matrix(a: &mut [f64], n: usize, p: usize, q: usize, c: f64, s: f64) {
    let a_pp = a[p * n + p];
    let a_qq = a[q * n + q];
    let a_pq = a[p * n + q];

    // Update diagonal elements
    a[p * n + p] = c * c * a_pp - 2.0 * s * c * a_pq + s * s * a_qq;
    a[q * n + q] = s * s * a_pp + 2.0 * s * c * a_pq + c * c * a_qq;

    // Zero out the (p,q) and (q,p) entries
    a[p * n + q] = 0.0;
    a[q * n + p] = 0.0;

    // Update remaining rows/columns
    for r in 0..n {
        if r == p || r == q {
            continue;
        }
        let a_rp = a[r * n + p];
        let a_rq = a[r * n + q];
        let new_rp = c * a_rp - s * a_rq;
        let new_rq = s * a_rp + c * a_rq;
        a[r * n + p] = new_rp;
        a[p * n + r] = new_rp;
        a[r * n + q] = new_rq;
        a[q * n + r] = new_rq;
    }
}

/// Apply a Jacobi (Givens) rotation to the eigenvector matrix V.
fn apply_jacobi_rotation_vectors(v: &mut [f64], n: usize, p: usize, q: usize, c: f64, s: f64) {
    for r in 0..n {
        let v_rp = v[r * n + p];
        let v_rq = v[r * n + q];
        v[r * n + p] = c * v_rp - s * v_rq;
        v[r * n + q] = s * v_rp + c * v_rq;
    }
}
