#![cfg(feature = "lapack")]

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::new_modules::matrix_decomp::svd;
use num_traits::{Float, Zero};
use std::fmt::Debug;

/// Type alias for complex least squares return type
/// Returns (solution, residuals, rank, singular_values)
type LstsqResult<T> = Result<(
    Array<T>,
    Array<T>, // Residuals are same type as matrix elements
    usize,
    Array<T>, // Singular values are same type as matrix elements
)>;

/// Compute the condition number of a matrix using the SVD method
///
/// The condition number is the ratio of the largest to smallest singular value
/// and provides a measure of how sensitive matrix operations are to numerical errors.
/// A high condition number indicates an ill-conditioned matrix.
///
/// This implementation includes various numerical stability enhancements:
/// 1. Proper handling of very small singular values
/// 2. Scaling to avoid overflow in calculations
/// 3. Classification of different condition number ranges
pub fn condition_number<T>(a: &Array<T>) -> Result<T>
where
    T: Float + Clone + Debug,
{
    // SVD is the most numerically stable method for computing condition number
    let (_, s, _) = svd(a)?;

    // Convert to vector for easier manipulation
    let s_vec = s.to_vec();

    // Find the largest and smallest singular values
    if s_vec.is_empty() {
        return Err(NumRs2Error::ComputationError(
            "Cannot compute condition number of empty matrix".to_string(),
        ));
    }

    // Get the largest singular value
    let max_sv = s_vec
        .iter()
        .cloned()
        .fold(T::zero(), |a, b| if a > b { a } else { b });

    // Get the smallest non-zero singular value
    // We use a threshold based on machine epsilon to determine effective zeros
    let eps = T::epsilon();
    let threshold = max_sv
        * eps
        * T::from(std::cmp::max(a.shape()[0], a.shape()[1])).unwrap_or_else(|| T::one());

    // Filter out singular values effectively zero
    let non_zero_sv = s_vec
        .iter()
        .cloned()
        .filter(|&sv| sv > threshold)
        .collect::<Vec<_>>();

    if non_zero_sv.is_empty() || max_sv == T::zero() {
        // Matrix is numerically singular (all singular values effectively zero)
        return Ok(T::infinity());
    }

    // Also check if there are singular values that are almost zero
    let min_sv_all = s_vec
        .iter()
        .cloned()
        .fold(max_sv, |a, b| if a < b { a } else { b });

    // If the ratio between largest and smallest is very large, return infinity
    if max_sv / min_sv_all > T::from(1e16).unwrap_or_else(|| T::max_value()) {
        return Ok(T::infinity());
    }

    // Get the smallest non-zero singular value
    let min_sv = non_zero_sv
        .iter()
        .cloned()
        .fold(max_sv, |a, b| if a < b { a } else { b });

    // Compute the condition number as the ratio of largest to smallest singular values
    let cond = max_sv / min_sv;

    // Check for overflow and handle appropriately
    if cond.is_infinite() || cond.is_nan() {
        // If we get overflow or NaN, return a high but finite condition number
        return Ok(T::max_value());
    }

    Ok(cond)
}

/// Calculate the reciprocal condition number, which is more numerically stable
/// for very ill-conditioned matrices where the ratio might overflow.
///
/// Returns a value between 0 and 1, where values close to 0 indicate ill-conditioning,
/// and values close to 1 indicate good conditioning.
pub fn rcond<T>(a: &Array<T>) -> Result<T>
where
    T: Float + Clone + Debug,
{
    let cond = condition_number(a)?;

    // Compute the reciprocal, handling potential underflow
    if cond.is_infinite() {
        Ok(T::zero())
    } else {
        Ok(T::one() / cond)
    }
}

/// Compute the sign and (natural) logarithm of the determinant of a matrix.
///
/// This is a more numerically stable way to compute the determinant for large matrices
/// where the determinant might overflow or underflow. Returns a tuple (sign, logdet)
/// where sign is -1, 0, or 1, and logdet is the natural logarithm of the absolute
/// value of the determinant.
///
/// # Arguments
/// * `a` - Input square matrix
///
/// # Returns
/// A tuple (sign, logdet) where:
/// - sign: The sign of the determinant (-1, 0, or 1)
/// - logdet: The natural logarithm of the absolute value of the determinant
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::new_modules::matrix_decomp::condition::slogdet;
///
/// let a = Array::from_vec(vec![2.0, 0.0, 0.0, 3.0]).reshape(&[2, 2]);
/// let (sign, logdet) = slogdet(&a).expect("slogdet should succeed for square matrix");
/// // For this diagonal matrix: det = 2*3 = 6, so sign = 1, logdet = ln(6) ≈ 1.79
/// ```
pub fn slogdet<T>(a: &Array<T>) -> Result<(i8, T)>
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
    // Check if the matrix is square
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "slogdet requires a square matrix".to_string(),
        ));
    }

    let n = shape[0];

    // For very small matrices, compute determinant directly
    if n <= 3 {
        let det = a.det()?;

        if det == T::zero() {
            return Ok((0, T::neg_infinity()));
        }

        let sign = if det > T::zero() { 1 } else { -1 };

        let logdet = num_traits::Float::ln(num_traits::Float::abs(det));
        return Ok((sign, logdet));
    }

    // For larger matrices, use LU decomposition which is more numerically stable
    let lu_result = crate::new_modules::matrix_decomp::lu::lu(a);

    match lu_result {
        Ok((_l, u, p)) => {
            // For LU decomposition PA = LU:
            // det(A) = det(P^-1) * det(L) * det(U)
            // det(L) = 1 (since L has 1's on diagonal)
            // det(U) = product of diagonal elements
            // det(P^-1) = det(P) = (-1)^(number of swaps)

            // Count the number of swaps in the permutation
            let p_vec = p.to_vec();
            let mut visited = vec![false; n];
            let mut num_swaps = 0;

            for i in 0..n {
                if !visited[i] {
                    let mut j = i;
                    let mut cycle_length = 0;

                    while !visited[j] {
                        visited[j] = true;
                        j = p_vec[j];
                        cycle_length += 1;
                    }

                    // A cycle of length k requires k-1 swaps
                    if cycle_length > 1 {
                        num_swaps += cycle_length - 1;
                    }
                }
            }

            // Calculate the sign from permutation
            let mut sign = if num_swaps % 2 == 0 { 1i8 } else { -1i8 };

            // Calculate log absolute value of determinant from U's diagonal
            let mut logdet = T::zero();

            for i in 0..n {
                let u_diag = u.get(&[i, i])?;

                if u_diag == T::zero() {
                    // Matrix is singular
                    return Ok((0, T::neg_infinity()));
                }

                // Update sign if diagonal element is negative
                if u_diag < T::zero() {
                    sign = -sign;
                }

                // Add log of absolute value
                logdet += num_traits::Float::ln(num_traits::Float::abs(u_diag));
            }

            Ok((sign, logdet))
        }
        Err(_) => {
            // If LU decomposition fails, fall back to SVD
            let (_, s, _) = svd(a)?;
            let s_vec = s.to_vec();

            // Check if any singular value is effectively zero
            let eps = T::epsilon();
            let threshold = eps
                * T::from(n).unwrap_or_else(|| T::one())
                * s_vec
                    .iter()
                    .cloned()
                    .fold(T::zero(), |a, b| if a > b { a } else { b });

            let mut logdet = T::zero();
            let mut zero_count = 0;

            for &sv in &s_vec {
                if sv <= threshold {
                    zero_count += 1;
                } else {
                    logdet += num_traits::Float::ln(sv);
                }
            }

            if zero_count > 0 {
                // Matrix is singular or numerically singular
                Ok((0, T::neg_infinity()))
            } else {
                // Singular values only carry the magnitude of the determinant, so the
                // sign cannot be recovered from `s` alone. Recover the true sign with an
                // independent LU factorization with partial pivoting:
                //   sign(det) = (parity of the row permutation)
                //               * (product of signs of the U diagonal pivots)
                // `logdet` (= Σ ln(σ_i)) already holds the correct magnitude.
                let sign = determinant_sign_via_lu(a)?;
                Ok((sign, logdet))
            }
        }
    }
}

/// Compute only the sign (-1, 0, or 1) of the determinant of a square matrix via
/// LU decomposition with partial pivoting.
///
/// This performs a self-contained Gaussian elimination with partial pivoting and
/// tracks two quantities:
/// 1. The parity of the row interchanges (each swap flips the sign).
/// 2. The signs of the diagonal pivots of the resulting upper-triangular factor.
///
/// The product of these gives `sign(det(A))`. Only the sign is returned; the
/// magnitude is intentionally ignored (callers obtain `log|det|` separately, e.g.
/// from the singular values), which keeps this routine free of overflow/underflow
/// concerns. A zero pivot indicates a singular matrix and yields a sign of `0`.
fn determinant_sign_via_lu<T>(a: &Array<T>) -> Result<i8>
where
    T: Float + Clone + Debug,
{
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "determinant sign computation requires a square matrix".to_string(),
        ));
    }

    let n = shape[0];

    // Working copy of the matrix as a dense row-major buffer.
    let mut m = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = a.get(&[i, j])?;
        }
    }

    // Tolerance for treating a pivot as numerically zero, scaled by the largest
    // matrix entry so the threshold is dimensionally consistent.
    let mut max_abs = T::zero();
    for value in &m {
        let abs_value = num_traits::Float::abs(*value);
        if abs_value > max_abs {
            max_abs = abs_value;
        }
    }
    let eps = T::epsilon();
    let pivot_threshold = if max_abs > T::zero() {
        eps * T::from(n).unwrap_or_else(|| T::one()) * max_abs
    } else {
        eps
    };

    // Sign accumulator: starts positive, flips on each row swap and each negative pivot.
    let mut sign: i8 = 1;

    for k in 0..n {
        // Partial pivoting: select the row with the largest magnitude in column k.
        let mut pivot_row = k;
        let mut pivot_val = num_traits::Float::abs(m[k * n + k]);
        for i in (k + 1)..n {
            let candidate = num_traits::Float::abs(m[i * n + k]);
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = i;
            }
        }

        // A zero column at this stage means the matrix is singular.
        if pivot_val <= pivot_threshold {
            return Ok(0);
        }

        // Swap rows if necessary, flipping the determinant sign each time.
        if pivot_row != k {
            for j in 0..n {
                m.swap(k * n + j, pivot_row * n + j);
            }
            sign = -sign;
        }

        // Account for the sign of this pivot (the corresponding U diagonal entry).
        if m[k * n + k] < T::zero() {
            sign = -sign;
        }

        // Eliminate entries below the pivot.
        let pivot = m[k * n + k];
        for i in (k + 1)..n {
            let factor = m[i * n + k] / pivot;
            if factor != T::zero() {
                for j in k..n {
                    let update = factor * m[k * n + j];
                    m[i * n + j] = m[i * n + j] - update;
                }
            }
        }
    }

    Ok(sign)
}

/// Solve a linear least-squares problem using SVD decomposition.
///
/// Computes the least-squares solution to the linear system Ax = b.
/// If the system is over-determined (more equations than unknowns), this
/// finds the solution that minimizes ||Ax - b||₂.
/// If the system is under-determined, this finds the minimum-norm solution.
///
/// # Arguments
/// * `a` - Coefficient matrix (m × n)
/// * `b` - Right-hand side vector or matrix (m × k)
/// * `rcond` - Cutoff for small singular values (relative to largest singular value).
///   Singular values smaller than rcond * largest_sv are set to zero.
///   If None, uses machine precision.
///
/// # Returns
/// A tuple (x, residuals, rank, singular_values) where:
/// - x: Least-squares solution(s)
/// - residuals: Sum of squared residuals (empty if m <= n or rank deficient)
/// - rank: Effective rank of matrix a
/// - singular_values: Singular values of a in descending order
///
/// # Examples
/// ```
/// use numrs2::prelude::*;
/// use numrs2::new_modules::matrix_decomp::condition::lstsq;
///
/// // Solve simple 2x2 system
/// let a = Array::from_vec(vec![1.0, 1.0, 1.0, 2.0]).reshape(&[2, 2]);
/// let b = Array::from_vec(vec![3.0, 4.0]);
/// let (x, residuals, rank, sv) = lstsq(&a, &b, None).expect("lstsq should succeed");
/// ```
pub fn lstsq<T>(a: &Array<T>, b: &Array<T>, rcond: Option<T>) -> LstsqResult<T>
where
    T: Float + Clone + Debug + 'static,
{
    // Check input dimensions
    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "lstsq requires a 2D coefficient matrix".to_string(),
        ));
    }

    let m = a_shape[0]; // number of equations
    let n = a_shape[1]; // number of unknowns

    // Check that b has the correct number of rows
    if b_shape[0] != m {
        return Err(NumRs2Error::ShapeMismatch {
            expected: vec![m],
            actual: b_shape.clone(),
        });
    }

    // Determine if b is a vector or matrix
    let k = if b_shape.len() == 1 { 1 } else { b_shape[1] };
    let b_is_vector = b_shape.len() == 1;

    // Compute SVD of A
    let (u, s, vt) = svd(a)?;
    let s_vec = s.to_vec();

    // Determine the effective rank using rcond
    let max_sv = s_vec
        .iter()
        .cloned()
        .fold(T::zero(), |a, b| if a > b { a } else { b });

    let cutoff = match rcond {
        Some(rc) => rc * max_sv,
        None => {
            let eps = T::epsilon();
            eps * T::from(std::cmp::max(m, n)).unwrap_or_else(|| T::one()) * max_sv
        }
    };

    let rank = s_vec.iter().filter(|&&sv| sv > cutoff).count();

    // Compute the pseudo-inverse solution: x = V * S^+ * U^T * b
    // where S^+ is the Moore-Penrose pseudo-inverse of the diagonal matrix S

    // Step 1: Compute U^T * b. `svd()` returns the FULL `U` (m x m -- see
    // oxiblas-lapack's `SvdDc::u()`, which is what backs it), so `ut_b`
    // always has `m` rows, regardless of how `m` compares to `n`.
    let ut_b = u.transpose().matmul(b)?;

    // Step 2: Apply S^+, landing directly in the `n x k` shape that `v`
    // (n x n) needs for the Step 3 matmul below. `S^+` is conceptually
    // `n x m` (the pseudo-inverse of the `m x n` rectangular "diagonal" S),
    // so `S^+ @ ut_b` always has `n` rows -- not `m`:
    //   - rows `0..min_dim` (`min_dim = min(m, n)`): `ut_b[i] / s_vec[i]`
    //     when `s_vec[i]` clears the cutoff, else zero -- same scaling the
    //     old code applied.
    //   - rows `min_dim..n` (non-empty only when `n > m`, the
    //     under-determined case): always zero -- `S^+` has no diagonal
    //     entry there, so nothing survives.
    //   - any `ut_b` rows at or past `min_dim` when `m > n` (the
    //     over-determined case) are dropped entirely -- `S^+` has no
    //     *column* there either, so they never contribute to `V @ S^+ @
    //     U^T @ b`.
    //
    // The old code instead started from `ut_b.clone()` (keeping all `m`
    // rows unconditionally) and only overwrote rows `0..min_dim` in place,
    // so it only ever matched `v`'s `n` columns when `m == n`. For every
    // non-square `a` the Step 3 `v.matmul(&s_pinv_ut_b)` below failed with
    // a shape mismatch -- this rebuild (into a fresh `n x k` buffer, rather
    // than a same-shape-as-`ut_b` in-place edit) is that fix.
    let min_dim = std::cmp::min(m, n);
    let mut s_pinv_ut_b_data = vec![T::zero(); n * k];
    for i in 0..min_dim {
        if i < s_vec.len() && s_vec[i] > cutoff {
            let sv_as_t = <T as num_traits::NumCast>::from(s_vec[i])
                .expect("singular value should convert to float type");
            for j in 0..k {
                s_pinv_ut_b_data[i * k + j] = ut_b.get(&[i, j])? / sv_as_t;
            }
        }
        // else: leave this row zero (small/negligible singular value) --
        // already the initial value from `vec![T::zero(); ..]` above.
    }
    let s_pinv_ut_b = Array::from_vec_shape(s_pinv_ut_b_data, &[n, k])?;

    // Step 3: Compute x = V * (S^+ * U^T * b)
    let v = vt.transpose();
    let x = v.matmul(&s_pinv_ut_b)?;

    // Reshape x if b was a vector
    let x_final = if b_is_vector && x.shape().len() == 2 && x.shape()[1] == 1 {
        let x_vec: Vec<T> = (0..x.shape()[0])
            .map(|i| x.get(&[i, 0]).expect("index should be valid"))
            .collect();
        Array::from_vec(x_vec)
    } else {
        x
    };

    // Compute residuals if the system is over-determined and full rank
    let residuals = if m > n && rank == n {
        // Compute ||Ax - b||²
        let ax = a.matmul(&x_final)?;
        let diff = if b_is_vector {
            // NOTE: `ax` is NOT a 1-D `[m]` array here, even though
            // `x_final` (matmul's second operand) is 1-D `[n]`: `Array::
            // matmul`'s "both operands already 2-D" fast path requires
            // *both* sides to already be 2-D, and only `x_final` is. So it
            // falls through to the general broadcasting path instead,
            // which promotes `x_final` to `[n, 1]` and returns the product
            // at that shape too, `[m, 1]` -- 2-D, with no squeeze back
            // down. Read `ax` at `[i, 0]` accordingly, not `[i]`.
            let ax_vec: Vec<T> = (0..ax.shape()[0])
                .map(|i| ax.get(&[i, 0]).expect("index should be valid"))
                .collect();
            let b_vec = b.to_vec();
            let diff_vec: Vec<T> = ax_vec
                .iter()
                .zip(b_vec.iter())
                .map(|(&ax_val, &b_val)| ax_val - b_val)
                .collect();
            Array::from_vec(diff_vec)
        } else {
            // Built from scratch (every entry is written exactly once), so a
            // flat Vec + `from_vec_shape` replaces `Array::zeros(..)` +
            // per-element `set()`: no wasted zero-fill and no per-element
            // unshare check.
            let mut diff_vec = Vec::with_capacity(b.shape()[0] * k);
            for i in 0..b.shape()[0] {
                for j in 0..k {
                    let ax_val = ax.get(&[i, j])?;
                    let b_val = b.get(&[i, j])?;
                    diff_vec.push(ax_val - b_val);
                }
            }
            Array::from_vec_shape(diff_vec, &b.shape())?
        };

        // Compute sum of squares for each column
        let mut residuals_vec = Vec::with_capacity(k);
        for j in 0..k {
            let mut sum_sq = T::zero();
            for i in 0..m {
                let val = if b_is_vector && k == 1 {
                    diff.get(&[i])?
                } else {
                    diff.get(&[i, j])?
                };
                sum_sq = sum_sq + val * val;
            }
            residuals_vec.push(sum_sq);
        }

        // Both branches were identical, so no condition needed
        Array::from_vec(residuals_vec)
    } else {
        // No residuals for under-determined or rank-deficient systems
        Array::from_vec(vec![])
    };

    Ok((x_final, residuals, rank, s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Regression coverage for the `lstsq` `s_pinv_ut_b` conversion (W3-A2):
    /// hoisted `array_mut()` instead of a per-`(i, j)` `Array::set`. Uses a
    /// MATRIX (not vector) right-hand side with `k = 2`, so the write loop
    /// actually iterates over more than one column.
    ///
    /// This is deliberately a SQUARE, invertible `a` (`m == n`), which was
    /// already reachable before the W4-D fix below. The non-square cases
    /// that fix unblocked -- `s_pinv_ut_b` needed truncating to `n` rows
    /// when `m > n` and zero-padding to `n` rows when `m < n`, since
    /// `svd()` returns the FULL `U` (m×m) and `Vᵀ` (n×n) (`SvdDc::u()`/
    /// `vt()`), while the old code always kept `ut_b`'s `m` rows -- have
    /// their own dedicated tests below (`lstsq_overdetermined_vector_
    /// matches_numpy`, `lstsq_underdetermined_vector_matches_numpy`,
    /// `lstsq_overdetermined_matrix_rhs_matches_numpy`), pinned against
    /// `np.linalg.lstsq`. The last of those also exercises the
    /// `diff_result` (matrix) residuals branch below, which this square
    /// test's `m == n` shape can never reach (residuals are only computed
    /// when `m > n`).
    ///
    /// Expected values are the closed-form solution `x = A⁻¹b`, computed by
    /// hand per column, independent of the implementation under test.
    #[test]
    fn lstsq_matrix_rhs_square_matches_closed_form_solve() {
        // A = [[1, 1], [1, 2]], det = 1, A^-1 = [[2, -1], [-1, 1]].
        let a = Array::from_vec(vec![1.0, 1.0, 1.0, 2.0]).reshape(&[2, 2]);
        // b has two columns: [3, 5] and [4, 6].
        let b = Array::from_vec(vec![3.0, 4.0, 5.0, 6.0]).reshape(&[2, 2]);

        let (x, _residuals, rank, _sv) =
            lstsq(&a, &b, None).expect("square full-rank lstsq should succeed");

        assert_eq!(rank, 2);
        assert_eq!(x.shape(), vec![2, 2]);

        // Column 0: A^-1 * [3, 5] = [2*3-1*5, -1*3+1*5] = [1, 2].
        assert_relative_eq!(x.get(&[0, 0]).expect("valid"), 1.0, epsilon = 1e-9);
        assert_relative_eq!(x.get(&[1, 0]).expect("valid"), 2.0, epsilon = 1e-9);
        // Column 1: A^-1 * [4, 6] = [2*4-1*6, -1*4+1*6] = [2, 2].
        assert_relative_eq!(x.get(&[0, 1]).expect("valid"), 2.0, epsilon = 1e-9);
        assert_relative_eq!(x.get(&[1, 1]).expect("valid"), 2.0, epsilon = 1e-9);

        // Cross-check independent of the hand-derived numbers: A*x must
        // reproduce b exactly for a square, well-conditioned, exact solve.
        let ax = a.matmul(&x).expect("matmul should succeed");
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    ax.get(&[i, j]).expect("valid"),
                    b.get(&[i, j]).expect("valid"),
                    epsilon = 1e-9
                );
            }
        }
    }

    /// Over-determined (`m > n`) least squares: the shape bug's primary
    /// repro. Before the fix, this errored on `v.matmul(&s_pinv_ut_b)`
    /// (`v` is `n x n`, but `s_pinv_ut_b` kept all `m` rows of `svd()`'s
    /// full `U`) for every non-square `a` -- this is the `m > n` half.
    ///
    /// Reference values from `numpy.linalg.lstsq` (numpy 2.4.2):
    /// `np.linalg.lstsq([[1,1],[1,2],[1,3],[1,4]], [6,5,7,10], rcond=None)`
    /// -> `x = [3.4999999999999987, 1.4]`, `res = [4.199999999999998]`,
    /// `rank = 2`.
    #[test]
    fn lstsq_overdetermined_vector_matches_numpy() {
        let a = Array::from_vec(vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).reshape(&[4, 2]);
        let b = Array::from_vec(vec![6.0, 5.0, 7.0, 10.0]);

        let (x, residuals, rank, _sv) =
            lstsq(&a, &b, None).expect("over-determined lstsq should now succeed");

        assert_eq!(rank, 2);
        assert_eq!(x.shape(), vec![2]);

        assert_relative_eq!(
            x.get(&[0]).expect("valid"),
            3.499_999_999_999_998_7,
            epsilon = 1e-9
        );
        assert_relative_eq!(x.get(&[1]).expect("valid"), 1.4, epsilon = 1e-9);

        // Defining property of the least-squares solution, independent of
        // the particular SVD/pinv algorithm used to reach it: the normal
        // equations A^T A x = A^T b hold up to floating-point error. `x`
        // comes back 1-D here (`b` was a vector) while `a.transpose()`'s
        // matmuls promote it to a column, so compare via flattened `Vec`s
        // rather than assuming a specific shape for either side.
        let at = a.transpose();
        let ata_x = at.matmul(&a).expect("matmul").matmul(&x).expect("matmul");
        let at_b = at.matmul(&b).expect("matmul");
        let ata_x_vec = ata_x.to_vec();
        let at_b_vec = at_b.to_vec();
        assert_eq!(ata_x_vec.len(), at_b_vec.len());
        for i in 0..ata_x_vec.len() {
            assert_relative_eq!(ata_x_vec[i], at_b_vec[i], epsilon = 1e-9);
        }

        // Residual: ||Ax - b||^2, pinned against NumPy's `res`. This is the
        // first test to reach the residuals computation at all (it needs
        // `m > n`, which errored unconditionally before this fix).
        assert_eq!(residuals.shape(), vec![1]);
        assert_relative_eq!(
            residuals.get(&[0]).expect("valid"),
            4.199_999_999_999_998,
            epsilon = 1e-9
        );
    }

    /// Under-determined (`m < n`) least squares -- NumPy's minimum-norm
    /// case, and the shape bug's other half: `s_pinv_ut_b` needs *padding*
    /// with `n - m` zero rows here, not truncation.
    ///
    /// Reference values from `numpy.linalg.lstsq` (numpy 2.4.2):
    /// `np.linalg.lstsq([[1,2,3],[4,5,6]], [7,8], rcond=None)` ->
    /// `x = [-3.055555555555555, 0.11111111111111172, 3.2777777777777772]`,
    /// `res = []`, `rank = 2`.
    #[test]
    fn lstsq_underdetermined_vector_matches_numpy() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        let b = Array::from_vec(vec![7.0, 8.0]);

        let (x, residuals, rank, _sv) =
            lstsq(&a, &b, None).expect("under-determined lstsq should now succeed");

        assert_eq!(rank, 2);
        assert_eq!(x.shape(), vec![3]);

        assert_relative_eq!(
            x.get(&[0]).expect("valid"),
            -3.055_555_555_555_555,
            epsilon = 1e-8
        );
        assert_relative_eq!(
            x.get(&[1]).expect("valid"),
            0.111_111_111_111_111_72,
            epsilon = 1e-8
        );
        assert_relative_eq!(
            x.get(&[2]).expect("valid"),
            3.277_777_777_777_777_2,
            epsilon = 1e-8
        );

        // A full-row-rank under-determined system has an *exact* solution
        // (there is nothing left to minimize): this is the identity that
        // actually pins down NumPy's minimum-norm convention here, unlike
        // the A^T A one above (which only characterizes least squares in
        // general, and would be satisfied by other x's too when `a` is
        // rank-deficient in columns the way it is here).
        let ax = a.matmul(&x).expect("matmul");
        let ax_vec = ax.to_vec();
        let b_vec = b.to_vec();
        assert_eq!(ax_vec.len(), b_vec.len());
        for i in 0..b_vec.len() {
            assert_relative_eq!(ax_vec[i], b_vec[i], epsilon = 1e-9);
        }

        // No residuals for an under-determined system: matches NumPy,
        // whose `res` is an empty array whenever `M <= N`.
        assert_eq!(residuals.shape(), vec![0]);
    }

    /// Matrix right-hand side, over-determined and full rank: exercises the
    /// `diff_result` (matrix, not vector) branch of the residuals
    /// computation above, which was unreachable before the shape fix --
    /// every over-determined call errored before residuals were ever
    /// computed, regardless of whether `b` was a vector or a matrix.
    ///
    /// Reference values from `numpy.linalg.lstsq` (numpy 2.4.2), same `A`
    /// as the over-determined vector test above, with
    /// `B = [[6,1,0],[5,2,1],[7,3,2],[10,4,5]]`.
    #[test]
    fn lstsq_overdetermined_matrix_rhs_matches_numpy() {
        let a = Array::from_vec(vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).reshape(&[4, 2]);
        #[rustfmt::skip]
        let b = Array::from_vec(vec![
            6.0, 1.0, 0.0,
            5.0, 2.0, 1.0,
            7.0, 3.0, 2.0,
            10.0, 4.0, 5.0,
        ])
        .reshape(&[4, 3]);

        let (x, residuals, rank, _sv) =
            lstsq(&a, &b, None).expect("over-determined matrix-RHS lstsq should now succeed");

        assert_eq!(rank, 2);
        assert_eq!(x.shape(), vec![2, 3]);

        #[rustfmt::skip]
        let expected_x = [
            [3.499_999_999_999_998_7,  2.428_503_789_574_373_6e-16, -1.999_999_999_999_999_3],
            [1.399_999_999_999_999_9,  0.999_999_999_999_999_7,      1.599_999_999_999_999_6],
        ];
        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(
                    x.get(&[i, j]).expect("valid"),
                    expected_x[i][j],
                    epsilon = 1e-8
                );
            }
        }

        // Normal equations A^T A X = A^T B, column-by-column -- independent
        // of NumPy's specific pinv/SVD internals.
        let at = a.transpose();
        let ata_x = at.matmul(&a).expect("matmul").matmul(&x).expect("matmul");
        let at_b = at.matmul(&b).expect("matmul");
        let ata_x_vec = ata_x.to_vec();
        let at_b_vec = at_b.to_vec();
        assert_eq!(ata_x_vec.len(), at_b_vec.len());
        for i in 0..ata_x_vec.len() {
            assert_relative_eq!(ata_x_vec[i], at_b_vec[i], epsilon = 1e-8);
        }

        // Residuals: one value per column of B, from the previously-dead
        // `diff_result` branch. The middle column's residual is ~0 (that
        // column of `B` lies exactly in `A`'s column space) -- `epsilon`
        // in `approx`'s `assert_relative_eq!` is also the absolute-error
        // fallback used whenever either side is (near) zero, so the same
        // tolerance covers it without a separate branch.
        assert_eq!(residuals.shape(), vec![3]);
        let expected_res = [4.199_999_999_999_998, 0.0, 1.2];
        for j in 0..3 {
            assert_relative_eq!(
                residuals.get(&[j]).expect("valid"),
                expected_res[j],
                epsilon = 1e-8
            );
        }
    }
}
