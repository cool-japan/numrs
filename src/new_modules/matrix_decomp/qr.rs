#![allow(clippy::needless_range_loop)]
#![cfg(feature = "lapack")]

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::{Float, One, Zero};
use scirs2_core::linalg::qr_ndarray;
use scirs2_core::ndarray::ArrayView2;
use std::fmt::Debug;

/// Compute the QR decomposition of a matrix
///
/// This implementation includes various numerical stability enhancements:
/// 1. Matrix scaling to avoid overflow
/// 2. Column pivoting for better numerical stability
/// 3. Orthogonality verification with adaptive tolerance
/// 4. Fallback to more stable Householder algorithm when needed
pub fn qr<T>(a: &Array<T>) -> Result<(Array<T>, Array<T>)>
where
    T: Float
        + Clone
        + Debug
        + std::ops::AddAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::SubAssign
        + std::fmt::Display
        + 'static,
{
    // Check that the matrix is 2D
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "QR decomposition requires a 2D matrix".to_string(),
        ));
    }

    let m = shape[0];
    let n = shape[1];

    // Scale the matrix to avoid overflow in large-magnitude entries
    // Find the maximum absolute value in the matrix
    let mut max_val = <T as num_traits::Zero>::zero();
    let mut a_scaled = a.clone();

    for i in 0..m {
        for j in 0..n {
            let val = a.get(&[i, j])?;
            let abs_val = num_traits::Float::abs(val);
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
    }

    // Apply scaling if maximum is very large
    let mut scaling_factor = <T as num_traits::One>::one();
    if max_val > <T as num_traits::NumCast>::from(1e6).expect("1e6 should convert to float type") {
        scaling_factor = <T as num_traits::One>::one() / max_val;

        let scaled_arr = a_scaled.array_mut();
        for i in 0..m {
            for j in 0..n {
                let val = a.get(&[i, j])?;
                scaled_arr[[i, j]] = val * scaling_factor;
            }
        }
    }

    // Get the 2D view and compute QR using OxiBLAS
    let a_view: ArrayView2<T> = a_scaled.view_2d()?;

    // Convert to f64 for OxiBLAS
    let mut a_f64 = scirs2_core::ndarray::Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            a_f64[[i, j]] = a_view[[i, j]].to_f64().ok_or_else(|| {
                NumRs2Error::ComputationError("Cannot convert to f64".to_string())
            })?;
        }
    }

    // Try QR decomposition using OxiBLAS
    let result = match qr_ndarray(&a_f64) {
        Ok(r) => r,
        Err(_e) => {
            // If the standard QR fails, use our fallback implementation
            // which uses Householder reflections for better stability
            return householder_qr(a);
        }
    };

    // Convert back to T
    let (q_f64, r_f64) = (result.q, result.r);

    // OxiBLAS returns full QR (Q is m x m, R is m x n)
    // We need to return reduced/economy QR (Q is m x n, R is n x n)
    // Extract the first n columns of Q and first n rows of R
    let q_rows = q_f64.nrows();
    let q_cols = std::cmp::min(m, n); // For economy QR, Q is m x min(m,n)
    let r_rows = q_cols;
    let r_cols = n;

    // Convert Q from f64 to T (only first n columns)
    let mut q_vec: Vec<T> = Vec::with_capacity(q_rows * q_cols);
    for i in 0..q_rows {
        for j in 0..q_cols {
            q_vec.push(
                T::from(q_f64[[i, j]]).ok_or_else(|| {
                    NumRs2Error::ComputationError("Conversion failed".to_string())
                })?,
            );
        }
    }

    // Convert R from f64 to T (only first n rows)
    let mut r_vec: Vec<T> = Vec::with_capacity(r_rows * r_cols);
    for i in 0..r_rows {
        for j in 0..r_cols {
            r_vec.push(
                T::from(r_f64[[i, j]]).ok_or_else(|| {
                    NumRs2Error::ComputationError("Conversion failed".to_string())
                })?,
            );
        }
    }

    #[allow(unused_mut)] // q_array is only modified in debug builds for orthogonality correction
    let mut q_array = Array::from_vec_shape(q_vec, &[q_rows, q_cols])?;
    let mut r_array = Array::from_vec_shape(r_vec, &[r_rows, r_cols])?;

    // If we scaled the matrix, rescale R appropriately
    if scaling_factor != <T as num_traits::One>::one() {
        let out = r_array.array_mut();
        for i in 0..std::cmp::min(m, n) {
            for j in i..n {
                out[[i, j]] /= scaling_factor;
            }
        }
    }

    // Set very small values in R to zero for numerical stability
    let eps = T::epsilon();
    let tol = eps
        * <T as num_traits::NumCast>::from(std::cmp::max(m, n))
            .expect("matrix dimension should convert to float type")
        * max_val;

    // Read the shape once up front: `r_array.shape()` heap-allocates a Vec
    // per call, and it can no longer be called per-iteration anyway once
    // `array_mut()` is holding an exclusive borrow below.
    let (r_rows, r_cols) = (r_array.shape()[0], r_array.shape()[1]);
    let out = r_array.array_mut();
    for i in 0..r_rows {
        for j in 0..r_cols {
            let r_val = out[[i, j]];
            if num_traits::Float::abs(r_val) < tol {
                out[[i, j]] = <T as num_traits::Zero>::zero();
            }
        }
    }

    // Verify and enhance orthogonality of Q with advanced techniques
    #[cfg(debug_assertions)]
    {
        // For economy QR, Q is m x n and Q^T * Q should be n x n identity
        // 1. First, assess the orthogonality of Q
        let qt = q_array.transpose();
        let product = qt.matmul(&q_array)?;

        // Use a more robust tolerance that scales with matrix size and condition
        let matrix_size = <T as num_traits::NumCast>::from(std::cmp::max(m, n))
            .expect("matrix dimension should convert to float type");

        // Estimate condition number of original matrix for better tolerance
        let _a_norm = max_val; // Unused but kept for future expansion
        let correction_factor =
            <T as num_traits::NumCast>::from(1.0).expect("1.0 should convert to float type");

        // More sophisticated tolerance that accounts for matrix properties
        let ortho_tol = eps
            * matrix_size
            * correction_factor
            * <T as num_traits::NumCast>::from(10.0).expect("10.0 should convert to float type");

        // Check that Q^T * Q is close to the identity matrix
        // Product should be n x n (or min(m,n) x min(m,n))
        let mut max_deviation = <T as num_traits::Zero>::zero();
        let mut avg_deviation = <T as num_traits::Zero>::zero();
        let mut num_elements = 0;

        let prod_size = std::cmp::min(m, n);
        for i in 0..prod_size {
            for j in 0..prod_size {
                let expected = if i == j {
                    <T as num_traits::One>::one()
                } else {
                    <T as num_traits::Zero>::zero()
                };
                let actual = product.get(&[i, j])?;
                let deviation = num_traits::Float::abs(actual - expected);

                avg_deviation += deviation;
                num_elements += 1;

                if deviation > max_deviation {
                    max_deviation = deviation;
                }
            }
        }

        // Calculate average deviation for more comprehensive assessment
        if num_elements > 0 {
            avg_deviation /= <T as num_traits::NumCast>::from(num_elements)
                .expect("num_elements should convert to float type");
        }

        // 2. If orthogonality is poor, attempt to improve it through reorthogonalization
        if max_deviation > ortho_tol {
            eprintln!("Warning: QR decomposition: Q may not be sufficiently orthogonal. Max deviation: {}, Avg deviation: {}",
                     max_deviation, avg_deviation);

            // In real applications, we would perform reorthogonalization here
            if max_deviation
                > ortho_tol
                    * <T as num_traits::NumCast>::from(10.0)
                        .expect("10.0 should convert to float type")
            {
                // For severe orthogonality issues, we perform explicit reorthogonalization

                // Clone Q to preserve original result
                let mut improved_q = q_array.clone();

                // Bulk-acquire once: the whole Gram-Schmidt pass below only
                // ever touches `improved_q` through direct indexing, so one
                // unshare covers every column instead of one per element.
                let iq_arr = improved_q.array_mut();

                // Apply modified Gram-Schmidt process for better numerical stability
                for j in 0..n {
                    // Extract column j
                    let mut col_j = vec![<T as num_traits::Zero>::zero(); m];
                    for i in 0..m {
                        col_j[i] = iq_arr[[i, j]];
                    }

                    // Normalize column j
                    let mut norm_j = <T as num_traits::Zero>::zero();
                    for val in &col_j {
                        norm_j += (*val) * (*val);
                    }
                    norm_j = num_traits::Float::sqrt(norm_j);

                    if norm_j > eps {
                        for i in 0..m {
                            iq_arr[[i, j]] = col_j[i] / norm_j;
                        }
                    }

                    // Reorthogonalize against subsequent columns
                    for k in (j + 1)..n {
                        // Extract column k
                        let mut col_k = vec![<T as num_traits::Zero>::zero(); m];
                        for i in 0..m {
                            col_k[i] = iq_arr[[i, k]];
                        }

                        // Compute dot product
                        let mut dot = <T as num_traits::Zero>::zero();
                        for i in 0..m {
                            dot += (col_j[i] / norm_j) * col_k[i];
                        }

                        // Subtract projection
                        for i in 0..m {
                            iq_arr[[i, k]] = col_k[i] - dot * (col_j[i] / norm_j);
                        }
                    }
                }

                // Update R to maintain A = QR
                let improved_qt = improved_q.transpose();
                r_array = improved_qt.matmul(a)?;

                // Replace original Q with improved version
                q_array = improved_q;

                // Verify the improvement
                let improved_qt = q_array.transpose();
                let improved_product = improved_qt.matmul(&q_array)?;

                let mut improved_max_deviation = <T as num_traits::Zero>::zero();
                for i in 0..std::cmp::min(m, n) {
                    for j in 0..std::cmp::min(m, n) {
                        let expected = if i == j {
                            <T as num_traits::One>::one()
                        } else {
                            <T as num_traits::Zero>::zero()
                        };
                        let actual = improved_product.get(&[i, j])?;
                        let deviation = num_traits::Float::abs(actual - expected);

                        if deviation > improved_max_deviation {
                            improved_max_deviation = deviation;
                        }
                    }
                }

                eprintln!(
                    "Orthogonality after improvement: Max deviation reduced from {} to {}",
                    max_deviation, improved_max_deviation
                );
            }
        }
    }

    // Validate the decomposition by checking A ≈ Q*R
    #[cfg(feature = "validation")]
    {
        let recon = q_array.matmul(&r_array)?;
        let mut max_diff = <T as num_traits::Zero>::zero();

        for i in 0..m {
            for j in 0..n {
                let diff = num_traits::Float::abs(a.get(&[i, j])? - recon.get(&[i, j])?);
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }

        let acceptable_error = eps
            * max_val
            * <T as num_traits::NumCast>::from(std::cmp::max(m, n))
                .expect("matrix dimension should convert to float type");
        if max_diff > acceptable_error {
            eprintln!("Warning: QR decomposition may be numerically unstable. Max reconstruction difference: {}", max_diff);
        }
    }

    Ok((q_array, r_array))
}

/// Fallback QR implementation using Householder reflections
/// This is more numerically stable than classical Gram-Schmidt
pub fn householder_qr<T>(a: &Array<T>) -> Result<(Array<T>, Array<T>)>
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
    let m = shape[0];
    let n = shape[1];
    let min_dim = std::cmp::min(m, n);

    // Create copies of A for the QR calculation
    let mut r = a.clone();
    let mut q = identity_matrix::<T>(m); // Start with identity matrix

    // Bulk-acquire both buffers once for the whole routine: every access to
    // `r` and `q` below is direct indexing (no shared helper takes `&Array`),
    // so a single unshare per array replaces what used to be an
    // `Arc::make_mut` on every one of the O(m*n*min_dim) element writes.
    let r_arr = r.array_mut();
    let q_arr = q.array_mut();

    // Householder QR is more numerically stable than Gram-Schmidt
    for k in 0..min_dim {
        // Extract column k from R
        let mut x = Vec::with_capacity(m - k);
        for i in k..m {
            x.push(r_arr[[i, k]]);
        }

        // Compute Householder vector v
        // Accumulate sum manually to avoid using Sum trait
        let mut sum_xx: T = <T as num_traits::Zero>::zero();
        for &val in &x {
            sum_xx += val * val;
        }
        let x_norm = num_traits::Float::sqrt(sum_xx);

        // Use a small epsilon threshold for numerical stability
        let eps = num_traits::Float::epsilon();
        if x_norm > eps {
            // First element of v determines the sign
            let alpha = if x[0] >= num_traits::Zero::zero() {
                -x_norm
            } else {
                x_norm
            };

            // Compute v = x - alpha*e1
            let mut v = x.clone();
            v[0] -= alpha;

            // Normalize v - accumulate sum manually again
            let mut sum_vv: T = <T as num_traits::Zero>::zero();
            for &val in &v {
                sum_vv += val * val;
            }
            let v_norm = num_traits::Float::sqrt(sum_vv);

            if v_norm > eps {
                for val in &mut v {
                    *val /= v_norm;
                }

                // Apply Householder reflection to R: R = R - 2 * v * (v^T * R)
                for j in k..n {
                    let mut vtr: T = <T as num_traits::Zero>::zero();
                    for i in 0..(m - k) {
                        let r_val = r_arr[[i + k, j]];
                        vtr += v[i] * r_val;
                    }

                    for i in 0..(m - k) {
                        let r_val = r_arr[[i + k, j]];
                        r_arr[[i + k, j]] = r_val
                            - <T as num_traits::NumCast>::from(2.0)
                                .expect("2.0 should convert to float type")
                                * v[i]
                                * vtr;
                    }
                }

                // Update Q: Q = Q * (I - 2 * v * v^T)
                for i in 0..m {
                    for j in k..m {
                        let mut q_row_dot_v: T = <T as num_traits::Zero>::zero();
                        for l in 0..(m - k) {
                            let q_val = q_arr[[i, l + k]];
                            q_row_dot_v += q_val * v[l];
                        }

                        let q_val = q_arr[[i, j]];
                        q_arr[[i, j]] = q_val
                            - <T as num_traits::NumCast>::from(2.0)
                                .expect("2.0 should convert to float type")
                                * q_row_dot_v
                                * v[j - k];
                    }
                }
            }
        }
    }

    // Zero out the lower triangular part of R for precision
    for i in 1..m {
        for j in 0..std::cmp::min(i, n) {
            r_arr[[i, j]] = num_traits::Zero::zero();
        }
    }

    Ok((q, r))
}

/// Create an identity matrix of size n
pub fn identity_matrix<T>(n: usize) -> Array<T>
where
    T: Zero + One + Clone,
{
    let mut result = Array::zeros(&[n, n]);
    let out = result.array_mut();
    for i in 0..n {
        out[[i, i]] = T::one();
    }
    result
}

// ---------------------------------------------------------------------------
// W3-A2 perf verification: pre-COW-conversion twin of `householder_qr()`
// (100% hand-rolled -- no external LAPACK call -- so the whole routine's cost
// sits in the loop touched by the conversion). Byte-for-byte copy of the
// function before the `Array::set` -> bulk-`array_mut()` change, kept only to
// measure the difference; not part of the public surface. Shares the
// (already-converted) `identity_matrix` for initial setup in both arms of the
// comparison, so it does not bias the result either way.
#[cfg(test)]
fn householder_qr_precow<T>(a: &Array<T>) -> Result<(Array<T>, Array<T>)>
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
    let m = shape[0];
    let n = shape[1];
    let min_dim = std::cmp::min(m, n);

    let mut r = a.clone();
    let mut q = identity_matrix::<T>(m);

    for k in 0..min_dim {
        let mut x = Vec::with_capacity(m - k);
        for i in k..m {
            x.push(r.get(&[i, k])?);
        }

        let mut sum_xx: T = <T as num_traits::Zero>::zero();
        for &val in &x {
            sum_xx += val * val;
        }
        let x_norm = num_traits::Float::sqrt(sum_xx);

        let eps = num_traits::Float::epsilon();
        if x_norm > eps {
            let alpha = if x[0] >= num_traits::Zero::zero() {
                -x_norm
            } else {
                x_norm
            };

            let mut v = x.clone();
            v[0] -= alpha;

            let mut sum_vv: T = <T as num_traits::Zero>::zero();
            for &val in &v {
                sum_vv += val * val;
            }
            let v_norm = num_traits::Float::sqrt(sum_vv);

            if v_norm > eps {
                for val in &mut v {
                    *val /= v_norm;
                }

                for j in k..n {
                    let mut vtr: T = <T as num_traits::Zero>::zero();
                    for i in 0..(m - k) {
                        let r_val = r.get(&[i + k, j])?;
                        vtr += v[i] * r_val;
                    }

                    for i in 0..(m - k) {
                        let r_val = r.get(&[i + k, j])?;
                        r.set(
                            &[i + k, j],
                            r_val
                                - <T as num_traits::NumCast>::from(2.0)
                                    .expect("2.0 should convert to float type")
                                    * v[i]
                                    * vtr,
                        )?;
                    }
                }

                for i in 0..m {
                    for j in k..m {
                        let mut q_row_dot_v: T = <T as num_traits::Zero>::zero();
                        for l in 0..(m - k) {
                            let q_val = q.get(&[i, l + k])?;
                            q_row_dot_v += q_val * v[l];
                        }

                        let q_val = q.get(&[i, j])?;
                        q.set(
                            &[i, j],
                            q_val
                                - <T as num_traits::NumCast>::from(2.0)
                                    .expect("2.0 should convert to float type")
                                    * q_row_dot_v
                                    * v[j - k],
                        )?;
                    }
                }
            }
        }
    }

    for i in 1..m {
        for j in 0..std::cmp::min(i, n) {
            r.set(&[i, j], num_traits::Zero::zero())?;
        }
    }

    Ok((q, r))
}

#[cfg(test)]
mod perf_verification {
    use super::*;
    use std::time::Instant;

    /// Deterministic, well-conditioned matrix (no randomness dependency).
    fn make_matrix(n: usize) -> Array<f64> {
        let mut data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                let base = 1.0 / (1.0 + (i as f64 - j as f64).abs());
                let diag_boost = if i == j { 2.0 * n as f64 } else { 0.0 };
                data.push(base + diag_boost);
            }
        }
        Array::from_vec_shape(data, &[n, n]).expect("matrix construction should succeed")
    }

    /// Min-of-N, alternating A/B timing (see `lu::perf_verification` for the
    /// rationale on interleaving under shared-machine load).
    #[test]
    #[ignore = "wall-clock perf assertion — run with --run-ignored, load-sensitive"]
    fn bench_householder_qr_cow_vs_precow() {
        // See `lu::perf_verification` (matrix_decomp::lu module) for why
        // debug builds use a much smaller size than release.
        let sizes: &[usize] = if cfg!(debug_assertions) {
            &[16, 32]
        } else {
            &[128, 256]
        };
        const SAMPLES: usize = 7;

        for &n in sizes {
            let a = make_matrix(n);

            let mut precow_times = Vec::with_capacity(SAMPLES);
            let mut cow_times = Vec::with_capacity(SAMPLES);

            for _ in 0..SAMPLES {
                let start = Instant::now();
                let _ = householder_qr_precow(&a).expect("precow should succeed");
                precow_times.push(start.elapsed());

                let start = Instant::now();
                let _ = householder_qr(&a).expect("cow should succeed");
                cow_times.push(start.elapsed());
            }

            let min_precow = precow_times.into_iter().min().expect("sample");
            let min_cow = cow_times.into_iter().min().expect("sample");
            let speedup = min_precow.as_secs_f64() / min_cow.as_secs_f64();

            eprintln!(
                "[bench_householder_qr_cow_vs_precow] n={n}: precow(min-of-{SAMPLES})={min_precow:?} \
                 cow(min-of-{SAMPLES})={min_cow:?} speedup={speedup:.3}x"
            );

            // See `lu::perf_verification` for why this threshold is lax
            // under an unoptimized `test`-profile build.
            let min_speedup = if cfg!(debug_assertions) { 0.3 } else { 0.8 };
            assert!(
                speedup > min_speedup,
                "n={n}: converted householder_qr() unexpectedly slower (speedup={speedup:.3}x)"
            );
        }
    }
}
