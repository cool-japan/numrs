#![cfg(feature = "lapack")]

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Internal f64 kernel — all heavy lifting is done here on plain Vec<Vec<f64>>
// so we avoid the overhead of the generic Array accessor calls inside tight
// loops.
// ---------------------------------------------------------------------------

/// Build an n×n identity matrix (plain f64 storage).
fn identity_f64(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// In-place matrix multiply C = A * B  (n×n, f64).
fn mat_mul_f64(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i][k];
            if a_ik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i][j] += a_ik * b[k][j];
            }
        }
    }
    c
}

/// Reduce A to upper Hessenberg form H = Q^T A Q via Householder reflectors.
/// Returns (H, Q) where Q accumulates all reflectors.
fn hessenberg_reduction(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut h: Vec<Vec<f64>> = a.to_vec();
    let mut q = identity_f64(n);

    for k in 0..n.saturating_sub(2) {
        // Build Householder vector v to zero h[k+2..n][k].
        let col_len = n - k - 1;
        let mut v: Vec<f64> = (0..col_len).map(|i| h[k + 1 + i][k]).collect();

        let mut sigma: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if sigma < 1e-300 {
            continue;
        }
        if v[0] >= 0.0 {
            sigma = -sigma;
        }
        v[0] -= sigma;
        let v_norm_sq: f64 = v.iter().map(|&x| x * x).sum();
        if v_norm_sq < 1e-300 {
            continue;
        }
        let two_over_vnorm = 2.0 / v_norm_sq;

        // Apply H from the left: h = P * h  (rows k+1..n, all cols)
        for j in 0..n {
            let dot: f64 = (0..col_len).map(|i| v[i] * h[k + 1 + i][j]).sum();
            let scale = two_over_vnorm * dot;
            for i in 0..col_len {
                h[k + 1 + i][j] -= scale * v[i];
            }
        }

        // Apply H from the right: h = h * P  (all rows, cols k+1..n)
        for i in 0..n {
            let dot: f64 = (0..col_len).map(|j| h[i][k + 1 + j] * v[j]).sum();
            let scale = two_over_vnorm * dot;
            for j in 0..col_len {
                h[i][k + 1 + j] -= scale * v[j];
            }
        }

        // Accumulate Q: q = q * P
        for i in 0..n {
            let dot: f64 = (0..col_len).map(|j| q[i][k + 1 + j] * v[j]).sum();
            let scale = two_over_vnorm * dot;
            for j in 0..col_len {
                q[i][k + 1 + j] -= scale * v[j];
            }
        }

        // Zero below the subdiagonal for cleanliness
        for i in (k + 2)..n {
            h[i][k] = 0.0;
        }
    }

    (h, q)
}

/// Apply a Givens rotation G_{p,q}(c,s) from the left to rows p and q of h
/// for columns col_start..n.
#[inline]
fn givens_left(h: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64, col_start: usize, n: usize) {
    for j in col_start..n {
        let hp = h[p][j];
        let hq = h[q][j];
        h[p][j] = c * hp + s * hq;
        h[q][j] = -s * hp + c * hq;
    }
}

/// Apply a Givens rotation G_{p,q}(c,s) from the right to columns p and q of h
/// for rows 0..row_end.
#[inline]
fn givens_right(h: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64, row_end: usize) {
    for i in 0..row_end {
        let hp = h[i][p];
        let hq = h[i][q];
        h[i][p] = c * hp + s * hq;
        h[i][q] = -s * hp + c * hq;
    }
}

/// Apply a Givens rotation from the right to columns p and q of q_mat
/// for rows 0..n (accumulate into Z).
#[inline]
fn givens_right_full(q_mat: &mut [Vec<f64>], p: usize, q_idx: usize, c: f64, s: f64, n: usize) {
    for i in 0..n {
        let zp = q_mat[i][p];
        let zq = q_mat[i][q_idx];
        q_mat[i][p] = c * zp + s * zq;
        q_mat[i][q_idx] = -s * zp + c * zq;
    }
}

/// Compute (c, s) for a Givens rotation that zeroes b:  [c s; -s c] [a; b] = [r; 0].
#[inline]
fn givens_cs(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        return (1.0, 0.0);
    }
    if a.abs() < b.abs() {
        let tau = -a / b;
        let s = 1.0 / (1.0 + tau * tau).sqrt();
        (s * tau, s)
    } else {
        let tau = -b / a;
        let c = 1.0 / (1.0 + tau * tau).sqrt();
        (c, c * tau)
    }
}

/// Perform a Francis double-shift QR step when the active submatrix is 2×2.
/// This is essentially a single-shift QR step using a Givens rotation.
fn francis_2x2_step(
    h: &mut [Vec<f64>],
    z: &mut [Vec<f64>],
    p: usize, // top of active 2×2 block (row/col index p and p+1)
    n: usize,
) {
    // Use Rayleigh-quotient shift from the bottom-right 1×1.
    let shift = h[p + 1][p + 1];
    let a = h[p][p] - shift;
    let b = h[p + 1][p];
    let (c, s) = givens_cs(a, b);

    givens_left(h, p, p + 1, c, s, p, n);
    givens_right(h, p, p + 1, c, s, p + 2);
    givens_right_full(z, p, p + 1, c, s, n);
}

/// Core Francis double-shift QR algorithm on an n×n upper Hessenberg matrix h.
/// On entry  h  is upper Hessenberg; z is the identity (will accumulate Q).
/// On exit   h  is in real Schur form (quasi-upper-triangular).
fn francis_qr(h: &mut [Vec<f64>], z: &mut [Vec<f64>]) {
    let n = h.len();
    if n <= 1 {
        return;
    }

    let eps = f64::EPSILON;
    let max_iters_per_eigenvalue = 30;
    let mut active = n; // The algorithm works on h[0..active][0..active]

    // Iteration count since last deflation (for exceptional shifts)
    let mut iter_count = 0usize;

    while active > 1 {
        // ----- Deflation check -----
        // Scan from bottom up for a small subdiagonal entry.
        let mut p = active - 1; // bottom of active window

        loop {
            if p == 0 {
                break;
            }
            let off = h[p][p - 1].abs();
            let tol = eps * (h[p - 1][p - 1].abs() + h[p][p].abs());
            if off <= tol {
                h[p][p - 1] = 0.0;
                break;
            }
            p -= 1;
        }

        if p == active - 1 {
            // Single eigenvalue deflated
            active -= 1;
            iter_count = 0;
            continue;
        }
        if p == active - 2 {
            // 2×2 block deflated
            active -= 2;
            iter_count = 0;
            continue;
        }

        // ----- Exceptional shift (Wilkinson ad-hoc) to avoid stagnation -----
        iter_count += 1;
        if iter_count > max_iters_per_eigenvalue * active {
            // Force convergence by applying an exceptional double shift
            // using the two smallest subdiagonal entries as proxies.
            let exceptional_shift =
                0.75 * h[active - 1][active - 2].abs() + h[active - 1][active - 1].abs();
            for i in 0..active {
                h[i][i] -= exceptional_shift;
            }
            // Single Givens step
            francis_2x2_step(h, z, active - 2, n);
            for i in 0..active {
                h[i][i] += exceptional_shift;
            }
            iter_count = 0;
            continue;
        }

        // ----- Standard Francis double-shift step -----
        if active - p >= 3 {
            // Apply the double-shift bulge-chasing step on h[p..active][p..active].
            // For simplicity we operate on the full h but only the active window
            // matters for the shift computation; the Householder applications cover
            // the right columns/rows automatically.

            // Temporarily restrict the view conceptually; we pass `active` as `m`.
            // The helper needs to know h starts at 0, so we shift the submatrix.
            // Rather than copying, we pass indices as offsets.
            francis_double_shift_step_range(h, z, p, active, n);
        } else {
            // active - p == 2: 2×2 block, use Givens
            francis_2x2_step(h, z, p, n);
        }
    }
}

/// Francis double-shift step on h[p_start..p_end][p_start..p_end],
/// chasing the bulge within that window, updating global h and z.
fn francis_double_shift_step_range(
    h: &mut [Vec<f64>],
    z: &mut [Vec<f64>],
    p_start: usize,
    p_end: usize,
    n: usize,
) {
    let m = p_end - p_start; // size of active window
    if m < 3 {
        return;
    }

    // Shifts from trailing 2×2 of the active window.
    let h11 = h[p_end - 2][p_end - 2];
    let h12 = h[p_end - 2][p_end - 1];
    let h21 = h[p_end - 1][p_end - 2];
    let h22 = h[p_end - 1][p_end - 1];

    let s = h11 + h22;
    let pp = h11 * h22 - h12 * h21;

    // First column of (H - s1*I)(H - s2*I) evaluated at the top-left of active window.
    let r0 = p_start;
    let r1 = p_start + 1;
    let r2 = p_start + 2;

    let x = h[r0][r0] * h[r0][r0] + h[r0][r1] * h[r1][r0] - s * h[r0][r0] + pp;
    let y = h[r1][r0] * (h[r0][r0] + h[r1][r1] - s);
    let z_val = h[r2][r1] * h[r1][r0];

    let mut v = [x, y, z_val];
    let sigma: f64 = {
        let sq: f64 = v.iter().map(|&a| a * a).sum::<f64>().sqrt();
        if v[0] >= 0.0 {
            -sq
        } else {
            sq
        }
    };
    if sigma.abs() < 1e-300 {
        return;
    }
    v[0] -= sigma;
    let v_norm_sq: f64 = v.iter().map(|&a| a * a).sum();
    if v_norm_sq < 1e-300 {
        return;
    }
    let two_over_v = 2.0 / v_norm_sq;

    // Apply from left: rows r0..r0+3, cols r0..n
    for j in r0..n {
        let dot = v[0] * h[r0][j] + v[1] * h[r1][j] + v[2] * h[r2][j];
        let scale = two_over_v * dot;
        h[r0][j] -= scale * v[0];
        h[r1][j] -= scale * v[1];
        h[r2][j] -= scale * v[2];
    }
    // Apply from right: cols r0..r0+3, rows 0..min(r0+4, n)
    {
        let row_end = std::cmp::min(r0 + 4, n);
        for i in 0..row_end {
            let dot = v[0] * h[i][r0] + v[1] * h[i][r1] + v[2] * h[i][r2];
            let scale = two_over_v * dot;
            h[i][r0] -= scale * v[0];
            h[i][r1] -= scale * v[1];
            h[i][r2] -= scale * v[2];
        }
    }
    // Accumulate into Z
    for i in 0..n {
        let dot = v[0] * z[i][r0] + v[1] * z[i][r1] + v[2] * z[i][r2];
        let scale = two_over_v * dot;
        z[i][r0] -= scale * v[0];
        z[i][r1] -= scale * v[1];
        z[i][r2] -= scale * v[2];
    }

    // Chase the bulge through the active window.
    for k in p_start..p_end.saturating_sub(2) {
        let nr = std::cmp::min(3, p_end - k);
        let c0 = k + 1;
        let c1 = k + 2;
        let c2 = k + 3;

        let mut vv: Vec<f64> = (0..nr).map(|i| h[k + 1 + i][k]).collect();
        let sigma_k: f64 = {
            let sq: f64 = vv.iter().map(|&a| a * a).sum::<f64>().sqrt();
            if vv[0] >= 0.0 {
                -sq
            } else {
                sq
            }
        };
        if sigma_k.abs() < 1e-300 {
            continue;
        }
        vv[0] -= sigma_k;
        let vv_norm_sq: f64 = vv.iter().map(|&a| a * a).sum();
        if vv_norm_sq < 1e-300 {
            continue;
        }
        let two_over_vv = 2.0 / vv_norm_sq;

        // Apply from left: rows k+1..k+1+nr, cols k..n
        for j in k..n {
            let dot: f64 = (0..nr).map(|i| vv[i] * h[k + 1 + i][j]).sum();
            let scale = two_over_vv * dot;
            for i in 0..nr {
                h[k + 1 + i][j] -= scale * vv[i];
            }
        }
        // Apply from right: rows 0..min(k+2+nr, n), cols k+1..k+1+nr
        {
            let row_top = std::cmp::min(k + 2 + nr, n);
            for i in 0..row_top {
                let dot: f64 = (0..nr).map(|j| h[i][k + 1 + j] * vv[j]).sum();
                let scale = two_over_vv * dot;
                for j in 0..nr {
                    h[i][k + 1 + j] -= scale * vv[j];
                }
            }
        }
        // Accumulate into Z
        for i in 0..n {
            let dot: f64 = (0..nr).map(|j| z[i][k + 1 + j] * vv[j]).sum();
            let scale = two_over_vv * dot;
            for j in 0..nr {
                z[i][k + 1 + j] -= scale * vv[j];
            }
        }
        // Zero subdiagonal fill-in (numerical clean-up)
        if nr == 3 {
            h[c2][k] = 0.0;
        }
        h[c1][k] = 0.0;
        let _ = (c0, c1, c2); // suppress unused warnings
    }
}

/// Extract eigenvalues from the real Schur form T.
/// 1×1 diagonal blocks → real eigenvalue.
/// 2×2 diagonal blocks → complex-conjugate pair (stored as two consecutive
///   entries in the returned Vec; the imaginary parts are ±im).
fn extract_eigenvalues_from_schur(t: &[Vec<f64>]) -> Vec<(f64, f64)> {
    let n = t.len();
    let mut eigenvalues = Vec::with_capacity(n);
    let eps = f64::EPSILON * 100.0;
    let mut i = 0;
    while i < n {
        if i + 1 < n && t[i + 1][i].abs() > eps * (t[i][i].abs() + t[i + 1][i + 1].abs()) {
            // 2×2 block — compute eigenvalues of [[a,b],[c,d]]
            let a = t[i][i];
            let b = t[i][i + 1];
            let c = t[i + 1][i];
            let d = t[i + 1][i + 1];
            let tr = a + d;
            let det = a * d - b * c;
            let disc = tr * tr - 4.0 * det;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                eigenvalues.push(((tr + sq) / 2.0, 0.0));
                eigenvalues.push(((tr - sq) / 2.0, 0.0));
            } else {
                let re = tr / 2.0;
                let im = (-disc).sqrt() / 2.0;
                eigenvalues.push((re, im));
                eigenvalues.push((re, -im));
            }
            i += 2;
        } else {
            eigenvalues.push((t[i][i], 0.0));
            i += 1;
        }
    }
    eigenvalues
}

// ---------------------------------------------------------------------------
// Generic Array<T> front-end  (preserves the existing public API)
// ---------------------------------------------------------------------------

/// Compute the Schur decomposition of a real matrix.
///
/// Returns `(Q, T)` where `A = Q * T * Q^T`, Q is orthogonal and T is in
/// real Schur form (upper quasi-triangular with 1×1 or 2×2 diagonal blocks).
///
/// # Algorithm
///
/// 1. Reduce A to upper Hessenberg form via Householder reflectors.
/// 2. Apply the Francis implicit double-shift QR algorithm with deflation
///    until the submatrix is fully reduced to Schur form.
/// 3. Extract eigenvalues from the 1×1 and 2×2 diagonal blocks.
///
/// This replaces the previous single-shift Rayleigh-quotient iteration which
/// failed to converge on matrices with complex-conjugate eigenpairs.
pub fn schur<T>(a: &Array<T>) -> Result<(Array<T>, Array<T>)>
where
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display,
{
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "Schur decomposition requires a square matrix".to_string(),
        ));
    }

    let n = shape[0];

    // Convert Array<T> to Vec<Vec<f64>> for the kernel.
    let mut a_f64: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            a_f64[i][j] = a.get(&[i, j])?.to_f64().ok_or_else(|| {
                NumRs2Error::ComputationError("Cannot convert matrix entry to f64".to_string())
            })?;
        }
    }

    // Step 1: Hessenberg reduction.
    let (mut h, mut q) = hessenberg_reduction(&a_f64);

    // Step 2: Francis double-shift QR.
    francis_qr(&mut h, &mut q);

    // Step 3: Clean up tiny sub-subdiagonal entries (more than one below diagonal).
    let eps = f64::EPSILON;
    for i in 0..n {
        for j in 0..n {
            if i > j + 1 {
                let tol = eps * (h[j][j].abs() + h[i][i].abs());
                if h[i][j].abs() <= tol {
                    h[i][j] = 0.0;
                }
            }
        }
    }

    // Convert back to Array<T>, built as flat row-major Vecs rather than
    // `Array::zeros(..)` + per-element `set()`: every entry is written
    // exactly once here, so `Array::from_vec_shape` both skips the zero-fill
    // `zeros()` would otherwise do just to have it immediately overwritten,
    // and needs no `Arc::make_mut` unshare check at all (the buffer is
    // fresh). One shared (i, j) loop still pushes into both Vecs in the
    // original interleaved order, so a conversion failure surfaces for the
    // same (i, j) and the same entry (Q before T) as before.
    let mut q_vec = Vec::with_capacity(n * n);
    let mut t_vec = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let q_val = T::from(q[i][j]).ok_or_else(|| {
                NumRs2Error::ComputationError("Cannot convert Q entry back to T".to_string())
            })?;
            q_vec.push(q_val);
            let t_val = T::from(h[i][j]).ok_or_else(|| {
                NumRs2Error::ComputationError("Cannot convert T entry back to T".to_string())
            })?;
            t_vec.push(t_val);
        }
    }
    let q_out = Array::from_vec_shape(q_vec, &[n, n])?;
    let t_out = Array::from_vec_shape(t_vec, &[n, n])?;

    Ok((q_out, t_out))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "lapack"))]
mod tests {
    use super::*;
    use crate::array::Array;

    /// Helper: multiply two n×n Array<f64> matrices.
    fn mat_mul_arr(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
        let n = a.shape()[0];
        let mut c = Array::zeros(&[n, n]);
        for i in 0..n {
            for k in 0..n {
                let a_ik = a.get(&[i, k]).unwrap_or(0.0);
                for j in 0..n {
                    let prev = c.get(&[i, j]).unwrap_or(0.0);
                    let b_kj = b.get(&[k, j]).unwrap_or(0.0);
                    let _ = c.set(&[i, j], prev + a_ik * b_kj);
                }
            }
        }
        c
    }

    /// Helper: max elementwise |product[i][j] - expected[i][j]| where expected = I.
    fn max_off_identity_arr(product: &Array<f64>) -> f64 {
        let n = product.shape()[0];
        let mut err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0_f64 } else { 0.0_f64 };
                let diff = (product.get(&[i, j]).unwrap_or(0.0) - expected).abs();
                if diff > err {
                    err = diff;
                }
            }
        }
        err
    }

    /// Test on a 4×4 matrix that has four distinct real eigenvalues.
    ///
    /// A = diag(1, 2, 3, 4) (already in Schur form) gives a trivial sanity
    /// check; we also try a rotation to make it more realistic.
    #[test]
    fn test_schur_real_eigenvalues() {
        // Construct a symmetric 4×4 matrix with known eigenvalues.
        // A = Q * diag(1, 2, 3, 4) * Q^T where Q is a 45° rotation in the (0,1) plane.
        let c = (std::f64::consts::PI / 4.0_f64).cos();
        let s = (std::f64::consts::PI / 4.0_f64).sin();
        // Q = [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        // diag = [1, 2, 3, 4]
        // A = Q diag Q^T
        let d = [1.0_f64, 2.0, 3.0, 4.0];
        let mut a_data = vec![0.0_f64; 16];
        // Build Q explicitly
        let mut q_data = [0.0_f64; 16];
        q_data[0] = c;
        q_data[1] = -s;
        q_data[1 * 4 + 0] = s;
        q_data[1 * 4 + 1] = c;
        q_data[2 * 4 + 2] = 1.0;
        q_data[3 * 4 + 3] = 1.0;

        // A = Q * diag * Q^T
        for i in 0..4 {
            for j in 0..4 {
                let mut sum = 0.0_f64;
                for k in 0..4 {
                    sum += q_data[i * 4 + k] * d[k] * q_data[j * 4 + k];
                }
                a_data[i * 4 + j] = sum;
            }
        }

        let a = Array::from_vec(a_data).reshape(&[4, 4]);
        let (q_out, t_out) = schur(&a).expect("Schur should succeed for real-eigenvalue matrix");

        // Verify T is quasi-upper-triangular: T[i][j] = 0 for i > j+1
        for i in 0..4_usize {
            for j in 0..4_usize {
                if i > j + 1 {
                    let val = t_out.get(&[i, j]).unwrap_or(0.0).abs();
                    assert!(
                        val < 1e-8,
                        "T[{},{}] = {:.3e} should be near zero (quasi-upper-triangular)",
                        i,
                        j,
                        val
                    );
                }
            }
        }

        // Verify Q is orthogonal: Q^T * Q ≈ I
        let qt = q_out.transpose();
        let qtq = mat_mul_arr(&qt, &q_out);
        let orth_err = max_off_identity_arr(&qtq);
        assert!(
            orth_err < 1e-8,
            "Q should be orthogonal; max error = {:.3e}",
            orth_err
        );

        // Verify A = Q * T * Q^T
        let qt2 = q_out.transpose();
        let qtq_t = mat_mul_arr(&mat_mul_arr(&q_out, &t_out), &qt2);
        for i in 0..4 {
            for j in 0..4 {
                let expected = a.get(&[i, j]).unwrap_or(0.0);
                let actual = qtq_t.get(&[i, j]).unwrap_or(0.0);
                assert!(
                    (actual - expected).abs() < 1e-8,
                    "Q*T*Q^T != A at ({},{}): expected {:.6}, got {:.6}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }

        // Collect eigenvalues from diagonal blocks and check they match {1,2,3,4}.
        let t_data: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..4).map(|j| t_out.get(&[i, j]).unwrap_or(0.0)).collect())
            .collect();
        let eigs = extract_eigenvalues_from_schur(&t_data);
        let mut real_eigs: Vec<f64> = eigs.iter().map(|&(re, _im)| re).collect();
        real_eigs.sort_by(|x, y| x.total_cmp(y));
        let expected_eigs = [1.0_f64, 2.0, 3.0, 4.0];
        for (actual, &exp) in real_eigs.iter().zip(expected_eigs.iter()) {
            assert!(
                (actual - exp).abs() < 1e-8,
                "Eigenvalue mismatch: expected {}, got {}",
                exp,
                actual
            );
        }
    }

    /// Test on a 4×4 matrix whose characteristic polynomial has two
    /// complex-conjugate eigenpairs.
    ///
    /// A = [[0, -1, 0, 0],
    ///      [1,  0, 0, 0],
    ///      [0,  0, 0, -2],
    ///      [0,  0, 2,  0]]
    ///
    /// This is block-diagonal: top-left block has eigenvalues ±i,
    /// bottom-right block has eigenvalues ±2i.
    #[test]
    fn test_schur_complex_eigenvalues() {
        let a = Array::from_vec(vec![
            0.0_f64, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.0,
        ])
        .reshape(&[4, 4]);

        let (q_out, t_out) = schur(&a).expect("Schur should succeed for complex-eigenvalue matrix");

        // Verify Q orthogonal
        let qt = q_out.transpose();
        let qtq = mat_mul_arr(&qt, &q_out);
        let orth_err = max_off_identity_arr(&qtq);
        assert!(
            orth_err < 1e-8,
            "Q should be orthogonal; max error = {:.3e}",
            orth_err
        );

        // Verify A = Q * T * Q^T
        let qt2 = q_out.transpose();
        let reconstructed = mat_mul_arr(&mat_mul_arr(&q_out, &t_out), &qt2);
        for i in 0..4 {
            for j in 0..4 {
                let expected = a.get(&[i, j]).unwrap_or(0.0);
                let actual = reconstructed.get(&[i, j]).unwrap_or(0.0);
                assert!(
                    (actual - expected).abs() < 1e-8,
                    "Q*T*Q^T != A at ({},{}): expected {:.6}, got {:.6}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }

        // Extract eigenvalues and check we have two complex-conjugate pairs.
        let t_data: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..4).map(|j| t_out.get(&[i, j]).unwrap_or(0.0)).collect())
            .collect();
        let eigs = extract_eigenvalues_from_schur(&t_data);
        assert_eq!(eigs.len(), 4, "Should have 4 eigenvalues");

        // The pairs should have |im| ≈ 1 and |im| ≈ 2 (in some order).
        let mut imag_parts: Vec<f64> = eigs.iter().map(|&(_re, im)| im.abs()).collect();
        imag_parts.sort_by(|x, y| x.total_cmp(y));

        // All real parts should be ≈ 0 (the rotation matrix has purely imaginary eigenvalues).
        for &(re, _im) in &eigs {
            assert!(
                re.abs() < 1e-8,
                "Real part of eigenvalue should be ≈ 0, got {}",
                re
            );
        }

        // Verify the 2×2 diagonal blocks have the expected trace (0) and determinant.
        // Block 1: trace = 0, det ≈ 1
        // Block 2: trace = 0, det ≈ 4
        // We check by verifying that the set {det_block1, det_block2} ≈ {1, 4}.
        let mut block_dets: Vec<f64> = Vec::new();
        let eps = 1e-8_f64;
        let mut k = 0;
        while k < 4 {
            if k + 1 < 4 && t_data[k + 1][k].abs() > eps {
                // 2×2 block
                let a2 = t_data[k][k];
                let b2 = t_data[k][k + 1];
                let c2 = t_data[k + 1][k];
                let d2 = t_data[k + 1][k + 1];
                let tr = a2 + d2;
                let det = a2 * d2 - b2 * c2;
                assert!(
                    tr.abs() < 1e-8,
                    "Trace of 2×2 block at ({},{}) should be ≈ 0, got {}",
                    k,
                    k,
                    tr
                );
                block_dets.push(det);
                k += 2;
            } else {
                k += 1;
            }
        }
        // We expect two 2×2 blocks with determinants ≈ 1 and ≈ 4.
        block_dets.sort_by(|x, y| x.total_cmp(y));
        assert_eq!(block_dets.len(), 2, "Should have exactly two 2×2 blocks");
        assert!(
            (block_dets[0] - 1.0).abs() < 1e-8,
            "First block det should be ≈ 1, got {}",
            block_dets[0]
        );
        assert!(
            (block_dets[1] - 4.0).abs() < 1e-8,
            "Second block det should be ≈ 4, got {}",
            block_dets[1]
        );
    }
}
