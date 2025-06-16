use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::new_modules::matrix_decomp::qr::{identity_matrix, qr};
use ndarray::ArrayView2;
use num_traits::Float;
use std::fmt::Debug;

/// Compute the Schur decomposition of a matrix
/// This returns (Q, T) where A = Q*T*Q^T, Q is orthogonal, and T is upper quasi-triangular
///
/// This implementation includes various numerical stability enhancements:
/// 1. Balancing for better conditioning
/// 2. Improved QR algorithm with shifts for faster convergence
/// 3. Convergence checks and iteration limits
pub fn schur<T>(a: &Array<T>) -> Result<(Array<T>, Array<T>)>
where
    T: Float + Clone + Debug + ndarray_linalg::Lapack,
{
    // Check if the matrix is square
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "Schur decomposition requires a square matrix".to_string(),
        ));
    }

    // Get 2D view of the array
    let _a_view: ArrayView2<T> = a.view_2d()?;

    // This would ideally use the LAPACK routine DGEES for real matrices
    // or ZGEES for complex matrices
    // Since ndarray-linalg doesn't directly expose these, we'll implement
    // a numerically stable QR algorithm with shifts

    let n = shape[0];

    // ---------- STEP 1: Balance the matrix ----------
    // Balancing improves the conditioning by scaling rows and columns
    let mut balanced = a.clone();
    let mut d = vec![num_traits::One::one(); n]; // Scaling factors

    // Compute scaling factors for balancing
    let max_iterations = 5; // Usually 5-10 iterations is enough
    let tol = num_traits::NumCast::from(0.95).unwrap(); // Convergence tolerance

    for _ in 0..max_iterations {
        let mut converged = true;

        for i in 0..n {
            // Compute row and column norms
            let mut row_sum: T = <T as num_traits::Zero>::zero();
            let mut col_sum: T = <T as num_traits::Zero>::zero();

            for j in 0..n {
                if i != j {
                    row_sum += num_traits::Float::abs(balanced.get(&[i, j])?);
                    col_sum += num_traits::Float::abs(balanced.get(&[j, i])?);
                }
            }

            if row_sum > <T as num_traits::Zero>::zero()
                && col_sum > <T as num_traits::Zero>::zero()
            {
                // Compute the scaling factor
                let f = num_traits::Float::sqrt(col_sum / row_sum);
                let s = if f > tol && f < <T as num_traits::One>::one() / tol {
                    f
                } else {
                    <T as num_traits::One>::one()
                };

                if s != <T as num_traits::One>::one() {
                    converged = false;

                    // Update the scaling factor
                    d[i] *= s;

                    // Scale the row
                    for j in 0..n {
                        let val = balanced.get(&[i, j])?;
                        balanced.set(&[i, j], val * s)?;
                    }

                    // Scale the column
                    for j in 0..n {
                        let val = balanced.get(&[j, i])?;
                        balanced.set(&[j, i], val / s)?;
                    }
                }
            }
        }

        if converged {
            break;
        }
    }

    // ---------- STEP 2: Reduce to upper Hessenberg form ----------
    // An upper Hessenberg matrix has zeros below the first subdiagonal
    // This significantly accelerates the QR algorithm
    let mut h = balanced.clone();
    let mut q = identity_matrix(n);

    for k in 0..(n - 2) {
        // Create Householder reflection to zero out elements below the subdiagonal
        let mut x = Vec::with_capacity(n - k - 1);
        for i in (k + 1)..n {
            x.push(h.get(&[i, k])?);
        }

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

                // Apply H from both sides to H: H = P*H*P
                // First, compute w = H * v
                for _j in k..n {
                    let mut w: Vec<T> = vec![<T as num_traits::Zero>::zero(); n - k - 1];

                    for i in 0..(n - k - 1) {
                        for l in 0..(n - k - 1) {
                            w[i] += h.get(&[i + k + 1, l + k + 1])? * v[l];
                        }
                    }

                    // Now update H = H - 2*v*w^T
                    for i in 0..(n - k - 1) {
                        for l in 0..(n - k - 1) {
                            let h_val = h.get(&[i + k + 1, l + k + 1])?;
                            h.set(
                                &[i + k + 1, l + k + 1],
                                h_val
                                    - <T as num_traits::NumCast>::from(2.0).unwrap() * v[i] * w[l],
                            )?;
                        }
                    }
                }

                // Update Q: Q = Q * P
                for i in 0..n {
                    let mut q_row_dot_v: T = num_traits::Zero::zero();
                    for l in 0..(n - k - 1) {
                        q_row_dot_v += q.get(&[i, l + k + 1])? * v[l];
                    }

                    for j in (k + 1)..n {
                        let q_val = q.get(&[i, j])?;
                        q.set(
                            &[i, j],
                            q_val
                                - <T as num_traits::NumCast>::from(2.0).unwrap()
                                    * q_row_dot_v
                                    * v[j - k - 1],
                        )?;
                    }
                }
            }
        }
    }

    // ---------- STEP 3: QR algorithm with double shifts ----------
    // We'll implement a simplified version for clarity
    // A full implementation would use double shifts and deflation
    let max_iterations = 50 * n; // Set a reasonable limit
    let mut iterations = 0;
    let tol = T::epsilon() * num_traits::NumCast::from(n * 10).unwrap();

    while iterations < max_iterations {
        // Check if we're done (H is already in Schur form)
        let mut done = true;
        for i in 0..(n - 1) {
            if num_traits::Float::abs(h.get(&[i + 1, i])?) > tol {
                done = false;
                break;
            }
        }

        if done {
            break;
        }

        // Apply one step of QR with a suitable shift
        // For simplicity, we'll use a basic Rayleigh quotient shift
        let shift = h.get(&[n - 1, n - 1])?;

        // Apply the shift
        for i in 0..n {
            let diag = h.get(&[i, i])?;
            h.set(&[i, i], diag - shift)?;
        }

        // Compute QR decomposition
        let (q_i, r_i) = qr(&h)?;

        // Update H = R*Q + shift*I and Q = Q*Q_i
        h = r_i.matmul(&q_i)?;

        // Add the shift back
        for i in 0..n {
            let diag = h.get(&[i, i])?;
            h.set(&[i, i], diag + shift)?;
        }

        // Update the accumulated Q
        q = q.matmul(&q_i)?;

        iterations += 1;
    }

    // ---------- STEP 4: Reapply balancing to get the final Q ----------
    // If we balanced the matrix initially, we need to adjust Q
    if d.iter().any(|&s| s != num_traits::One::one()) {
        for i in 0..n {
            for j in 0..n {
                let val = q.get(&[i, j])?;
                q.set(&[i, j], val * d[j] / d[i])?;
            }
        }
    }

    // The resulting H matrix should be approximately upper triangular (Schur form)
    // and Q should be orthogonal, satisfying A = Q * H * Q^T

    // Clean up tiny elements below main diagonal for precision
    for i in 1..n {
        for j in 0..(i - 1) {
            let val = h.get(&[i, j])?;
            if num_traits::Float::abs(val) < tol {
                h.set(&[i, j], num_traits::Zero::zero())?;
            }
        }
    }

    Ok((q, h))
}
