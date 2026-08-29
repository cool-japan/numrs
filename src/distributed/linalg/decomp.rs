//! Decomposition-based orchestration: algorithms that solve a problem by
//! composing the factorizations in [`mod@super::tsqr`] and [`super::cholesky`]
//! rather than factoring anything themselves.
//!
//! Everything here rests on the same observation. A tall matrix's whole
//! numerical content fits in the `n x n` factor `R` that [`mod@super::tsqr`]
//! produces: `A = Q R` with orthonormal `Q`, so `A` and `R` share their
//! singular values, their least-squares solutions and their conditioning.
//! Whatever a caller wants from `A`, the expensive part happens on a matrix
//! small enough for one rank to hold, and the distributed side of the work
//! is a single pass of `Q` in one direction or the other.
//!
//! - [`distributed_qr`] materializes `Q` by rotating the identity through
//!   the stored tree.
//! - [`distributed_svd`] diagonalizes `R` and carries the left factor back
//!   out: `A = Q (U_R S V^T) = (Q U_R) S V^T`.
//! - [`distributed_solve`] is a least-squares fit: `Q^T b` up the tree, one
//!   triangular solve against `R`, done.
//! - [`distributed_solve_spd`] is the odd one out and delegates to
//!   [`super::cholesky`], because a symmetric positive definite system wants
//!   half the arithmetic and a completely different data layout.
//!
//! # No SPD sniffing
//!
//! [`distributed_solve`] never inspects `A` to decide whether it could get
//! away with Cholesky. A symmetry test costs a full pass and a tolerance
//! nobody can pick correctly, and silently switching algorithms makes the
//! error behaviour of a call depend on the *values* a caller passed rather
//! than on the function they named. The two paths are two functions, and the
//! layouts they need ([`Layout::RowBlock`] versus
//! [`Layout::ColBlockCyclic`]) make the choice explicit at the call site
//! anyway.
//!
//! # Root-side steps fail collectively
//!
//! Diagonalizing `R` and solving against it happen on one rank. Both go out
//! through [`super::bcast_fallible_bytes`], so a numeric failure becomes the
//! same error on every rank instead of a hang.

use super::matrix::{decode_matrix, encode_matrix, DistFloat, DistributedMatrix, Layout};
use super::tsqr::{tsqr, TsqrFactorization, ROOT};
use super::{bcast_fallible_bytes, DistTransport, DistributedLinalgError};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

/// Tag for broadcasting the root's SVD of `R`.
const TAG_SVD_FACTORS: u64 = 0x5000;
/// Tag for broadcasting the root's least-squares solution.
const TAG_SOLVE_X: u64 = 0x5100;

/// Read one [`encode_matrix`] frame out of a concatenation of them,
/// advancing `cursor`.
///
/// The frame is self-describing (`rows`, `cols`, then the elements), so a
/// sequence of them needs no separate length table.
fn take_matrix<T: DistFloat>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<Array2<T>, DistributedLinalgError> {
    let truncated =
        || DistributedLinalgError::Transport("truncated multi-part matrix payload".to_string());
    let header = bytes.get(*cursor..*cursor + 16).ok_or_else(truncated)?;
    let rows_bytes: [u8; 8] = header
        .get(0..8)
        .ok_or_else(truncated)?
        .try_into()
        .map_err(|_| truncated())?;
    let cols_bytes: [u8; 8] = header
        .get(8..16)
        .ok_or_else(truncated)?
        .try_into()
        .map_err(|_| truncated())?;
    let rows = u64::from_le_bytes(rows_bytes) as usize;
    let cols = u64::from_le_bytes(cols_bytes) as usize;
    let len = 16
        + rows
            .checked_mul(cols)
            .and_then(|count| count.checked_mul(T::ELEM_BYTES))
            .ok_or_else(|| {
                DistributedLinalgError::Transport(format!(
                    "multi-part payload declares an implausible {rows}x{cols} part"
                ))
            })?;
    let frame = bytes.get(*cursor..*cursor + len).ok_or_else(truncated)?;
    *cursor += len;
    decode_matrix::<T>(frame)
}

/// Identity matrix of order `n`.
fn identity<T: DistFloat>(n: usize) -> Array2<T> {
    let mut eye = Array2::<T>::zeros((n, n));
    for i in 0..n {
        eye[[i, i]] = T::one();
    }
    eye
}

/// A local thin SVD: `U`, the singular values, and `V^T`.
type LocalSvd<T> = (Array2<T>, Vec<T>, Array2<T>);

/// A distributed thin SVD: `U` spread over the ranks, with the singular
/// values and `V^T` replicated.
pub type DistributedSvd<T> = (DistributedMatrix<T>, Vec<T>, Array2<T>);

/// Sweeps a one-sided Jacobi SVD may take before it is declared stuck.
/// Convergence is quadratic once the columns are nearly orthogonal, so a
/// well-formed problem finishes in under ten; this bound only exists so a
/// pathological input returns an error instead of spinning.
const JACOBI_MAX_SWEEPS: usize = 60;

/// Thin SVD `A = U diag(s) V^T` of a small dense `A` (`m >= n`), by
/// one-sided Jacobi.
///
/// # Why not `scirs2_linalg::svd`
///
/// That routine forms `A^T A` and eigendecomposes it. Squaring the matrix
/// squares its condition number, which costs half the available digits: its
/// factors carry roughly `sqrt(eps)` relative error, and reconstructing even
/// a benign, well-conditioned `32 x 4` factor through it misses by about
/// `6e-4`. That is serviceable for a rough spectrum and unusable here, where
/// this factorization *is* the result — [`distributed_svd`] hands every
/// caller a `U` built directly out of it.
///
/// One-sided Jacobi instead orthogonalizes `A`'s columns in place with plane
/// rotations and never forms a cross-product. It is backward stable, it
/// attains high *relative* accuracy even on badly graded matrices (where the
/// QR algorithm loses the small singular values entirely), and the
/// accumulated rotations give `V` directly. Its cost is `O(m n^2)` per sweep,
/// which is irrelevant precisely because TSQR exists to make this matrix
/// small: `n` is the narrow dimension, and this runs on one rank against an
/// `n x n` input.
fn jacobi_svd<T: DistFloat>(a: &ArrayView2<'_, T>) -> Result<LocalSvd<T>, DistributedLinalgError> {
    let (m, n) = a.dim();
    if m < n {
        return Err(DistributedLinalgError::UnsupportedShape(format!(
            "one-sided Jacobi needs at least as many rows as columns, got {m}x{n}"
        )));
    }
    if n == 0 {
        return Err(DistributedLinalgError::InvalidDimensions { rows: m, cols: n });
    }

    // Bring the magnitude to unit scale once, up front, the way LAPACK's
    // `dgesvj` does. The sweep works with sums of squares, so an input whose
    // entries reach ~1e19 in f32 (or ~1e154 in f64) overflows `alpha` to
    // infinity — and an infinite `alpha` reads as "already orthogonal",
    // ending the sweep immediately and returning a non-orthonormal U with
    // no error at all. Scaling costs one pass and rules the whole failure
    // class out; it also lifts denormal inputs away from underflow. `U` and
    // `V` are scale invariant, so only the singular values are scaled back.
    let scale = a.iter().fold(T::zero(), |acc, value| acc.max(value.abs()));
    if !scale.is_finite() {
        return Err(DistributedLinalgError::LinalgError(
            "cannot decompose a matrix containing non-finite entries".to_string(),
        ));
    }

    // `work` starts as A/scale and ends as U * diag(s/scale): each rotation
    // is applied to it and mirrored into `v`, so `work == (A/scale) * v`
    // holds throughout.
    let mut work = if scale > T::zero() {
        a.mapv(|value| value / scale)
    } else {
        a.to_owned()
    };
    let mut v = identity::<T>(n);
    let two = T::one() + T::one();

    let mut sweeps = 0usize;
    loop {
        let mut rotated = false;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = T::zero();
                let mut beta = T::zero();
                let mut gamma = T::zero();
                for i in 0..m {
                    let left = work[[i, p]];
                    let right = work[[i, q]];
                    alpha += left * left;
                    beta += right * right;
                    gamma += left * right;
                }
                // Already orthogonal (or one column is null): nothing to do.
                // The test is relative, so it does not chase rounding noise
                // on large columns nor stop early on small ones.
                if gamma == T::zero() || alpha == T::zero() || beta == T::zero() {
                    continue;
                }
                // Cauchy-Schwarz gives |gamma| <= sqrt(alpha) * sqrt(beta),
                // so this is the relative test — but taking each root
                // *separately* matters. `alpha` and `beta` are already sums
                // of squares, so `(alpha * beta)` is a fourth power of the
                // column norm and overflows at a norm of ~6e9 in f32 (an
                // ordinary magnitude for physical data). `eps * inf` is
                // `inf`, every pair would then compare as already
                // orthogonal, the sweep would end immediately, and the
                // function would return A's normalized columns with V = I:
                // a perfect reconstruction with a `U` that is not
                // orthonormal and an `s` that are not singular values.
                // Rooting first keeps every quantity at column-norm scale.
                // `householder::factor` scales for the same reason.
                if gamma.abs() <= T::epsilon() * alpha.sqrt() * beta.sqrt() {
                    continue;
                }
                rotated = true;

                // Diagonalize [[alpha, gamma], [gamma, beta]]. Taking the
                // smaller root of the tangent keeps the rotation under 45
                // degrees, which is what makes the sweep contract.
                let zeta = (beta - alpha) / (two * gamma);
                let tangent = if zeta >= T::zero() {
                    T::one() / (zeta + (T::one() + zeta * zeta).sqrt())
                } else {
                    -T::one() / (-zeta + (T::one() + zeta * zeta).sqrt())
                };
                let cosine = T::one() / (T::one() + tangent * tangent).sqrt();
                let sine = cosine * tangent;

                for i in 0..m {
                    let left = work[[i, p]];
                    let right = work[[i, q]];
                    work[[i, p]] = cosine * left - sine * right;
                    work[[i, q]] = sine * left + cosine * right;
                }
                for i in 0..n {
                    let left = v[[i, p]];
                    let right = v[[i, q]];
                    v[[i, p]] = cosine * left - sine * right;
                    v[[i, q]] = sine * left + cosine * right;
                }
            }
        }
        if !rotated {
            break;
        }
        sweeps += 1;
        if sweeps >= JACOBI_MAX_SWEEPS {
            return Err(DistributedLinalgError::ConvergenceFailed(sweeps));
        }
    }

    // The column norms are the singular values of the *scaled* matrix;
    // normalizing gives U, and undoing the entry scaling gives the singular
    // values of the original.
    let mut singular = Vec::with_capacity(n);
    for j in 0..n {
        let mut norm_squared = T::zero();
        for i in 0..m {
            let value = work[[i, j]];
            norm_squared += value * value;
        }
        singular.push(norm_squared.sqrt());
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        singular
            .get(j)
            .and_then(|right| singular.get(i).map(|left| (right, left)))
            .and_then(|(right, left)| right.partial_cmp(left))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut u = Array2::<T>::zeros((m, n));
    let mut vt = Array2::<T>::zeros((n, n));
    let mut sorted = vec![T::zero(); n];
    let mut settled = vec![false; n];
    for (slot, &source) in order.iter().enumerate() {
        let value = *singular
            .get(source)
            .ok_or_else(|| DistributedLinalgError::LinalgError("bad sort order".to_string()))?;
        // Undo the entry scaling here rather than on `work`: `u` is built by
        // dividing by the *unscaled* norm, so the two must not be mixed.
        sorted[slot] = value * scale;
        for i in 0..n {
            vt[[slot, i]] = v[[i, source]];
        }
        if value > T::zero() {
            for i in 0..m {
                u[[i, slot]] = work[[i, source]] / value;
            }
            settled[slot] = true;
        }
    }

    // A rank-deficient input leaves some columns of U undetermined — the
    // data says nothing about them. Fill them with directions orthogonal to
    // everything already fixed, so U is orthonormal whatever the rank,
    // instead of shipping zero columns that would quietly fail `U^T U = I`.
    for j in 0..n {
        if settled[j] {
            continue;
        }
        let mut best: Option<(T, Vec<T>)> = None;
        for candidate in 0..m {
            let mut trial = vec![T::zero(); m];
            trial[candidate] = T::one();
            // Twice: one pass can leave a direction that was nearly in the
            // settled span with nothing but rounding error.
            for _ in 0..2 {
                for prior in 0..n {
                    if !settled[prior] {
                        continue;
                    }
                    let mut projection = T::zero();
                    for i in 0..m {
                        projection += u[[i, prior]] * trial[i];
                    }
                    for i in 0..m {
                        trial[i] -= projection * u[[i, prior]];
                    }
                }
            }
            let mut norm_squared = T::zero();
            for value in &trial {
                norm_squared += *value * *value;
            }
            let norm = norm_squared.sqrt();
            if best.as_ref().is_none_or(|(previous, _)| norm > *previous) {
                best = Some((norm, trial));
            }
        }
        let (norm, trial) = best.ok_or_else(|| {
            DistributedLinalgError::LinalgError(
                "cannot complete an orthonormal basis for U".to_string(),
            )
        })?;
        if norm <= T::epsilon().sqrt() {
            return Err(DistributedLinalgError::LinalgError(format!(
                "cannot complete an orthonormal basis for U at column {j}"
            )));
        }
        for i in 0..m {
            u[[i, j]] = trial[i] / norm;
        }
        settled[j] = true;
    }

    Ok((u, sorted, vt))
}

/// Solve `R X = B` for an upper triangular `R`, by back substitution.
fn solve_upper_triangular<T: DistFloat>(
    r: &ArrayView2<'_, T>,
    b: &ArrayView2<'_, T>,
) -> Result<Array2<T>, DistributedLinalgError> {
    let n = r.nrows();
    if r.ncols() != n || b.nrows() != n {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "cannot solve a {:?} triangular system against a {:?} right-hand side",
            r.dim(),
            b.dim()
        )));
    }
    // Scale the singularity test by the factor's own magnitude: an absolute
    // threshold would call a well-conditioned but small-valued R singular.
    let scale = r.iter().fold(T::zero(), |acc, v| acc.max(v.abs()));
    let dimension = <T as num_traits::NumCast>::from(n).ok_or_else(|| {
        DistributedLinalgError::LinalgError(format!("cannot represent dimension {n} as a float"))
    })?;
    let tolerance = scale * dimension * T::epsilon();

    let mut x = b.to_owned();
    for column in 0..x.ncols() {
        for i in (0..n).rev() {
            let pivot = r[[i, i]];
            if pivot.abs() <= tolerance {
                return Err(DistributedLinalgError::SingularMatrix);
            }
            let mut acc = x[[i, column]];
            for j in (i + 1)..n {
                acc -= r[[i, j]] * x[[j, column]];
            }
            x[[i, column]] = acc / pivot;
        }
    }
    Ok(x)
}

/// Check that `b` is distributed the same way as `a` and has matching rows.
///
/// Both operands' global shapes and layouts are replicated facts, so this
/// rejects on every rank at once and cannot strand a peer mid-collective.
fn check_rhs<T: DistFloat>(
    a: &DistributedMatrix<T>,
    b: &DistributedMatrix<T>,
) -> Result<(), DistributedLinalgError> {
    if b.layout() != Layout::RowBlock {
        return Err(DistributedLinalgError::UnsupportedShape(format!(
            "the right-hand side must be in Layout::RowBlock, got {:?}",
            b.layout()
        )));
    }
    let (rows, _) = a.global_shape();
    let (b_rows, _) = b.global_shape();
    if rows != b_rows {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "A has {rows} rows but the right-hand side has {b_rows}"
        )));
    }
    Ok(())
}

/// Factor a row-block distributed `A` as `Q R`, materializing `Q`.
///
/// `Q` comes back distributed exactly like `A` (`m x n`, [`Layout::RowBlock`])
/// and `R` (`n x n`) replicated on every rank. Forming `Q` costs a full
/// downward pass of the tree, so callers who only need to *apply* it should
/// keep the [`TsqrFactorization`] from [`super::tsqr::tsqr`] instead.
pub async fn distributed_qr<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    comm: &C,
) -> Result<(DistributedMatrix<T>, Array2<T>), DistributedLinalgError> {
    let factorization = tsqr(a, comm).await?;
    let (rows, cols) = a.global_shape();
    let eye = identity::<T>(cols);
    let local = factorization.apply_q(comm, Some(eye.view())).await?;
    let q = DistributedMatrix::from_local(
        Layout::RowBlock,
        rows,
        cols,
        comm.rank(),
        comm.world_size(),
        local,
    )?;
    Ok((q, factorization.r().to_owned()))
}

/// Thin SVD `A = U diag(s) V^T` of a row-block distributed `A`.
///
/// TSQR first, then the root diagonalizes the `n x n` `R` and broadcasts the
/// three factors, then `U = Q U_R` rides back down the tree. `U` is
/// distributed like `A`; `s` and `V^T` are replicated.
///
/// The root computes `svd(R)` alone even though `R` is replicated: the
/// factors then agree bit for bit on every rank regardless of how the local
/// solver schedules its work, which a redundant per-rank call would not
/// guarantee.
pub async fn distributed_svd<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    comm: &C,
) -> Result<DistributedSvd<T>, DistributedLinalgError> {
    let factorization = tsqr(a, comm).await?;
    let (rows, cols) = a.global_shape();
    let ctx = comm.next_ctx();

    let produced = if comm.rank() == ROOT {
        svd_of_r(&factorization)
    } else {
        Ok(Vec::new())
    };
    let bytes = bcast_fallible_bytes(comm, ROOT, ctx, TAG_SVD_FACTORS, produced).await?;

    let mut cursor = 0usize;
    let u_r = take_matrix::<T>(&bytes, &mut cursor)?;
    let singular = take_matrix::<T>(&bytes, &mut cursor)?;
    let vt = take_matrix::<T>(&bytes, &mut cursor)?;
    if u_r.dim() != (cols, cols) || vt.dim() != (cols, cols) || singular.dim() != (cols, 1) {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "root broadcast a {:?} / {:?} / {:?} SVD of an {cols}x{cols} R",
            u_r.dim(),
            singular.dim(),
            vt.dim()
        )));
    }

    let local = factorization.apply_q(comm, Some(u_r.view())).await?;
    let u = DistributedMatrix::from_local(
        Layout::RowBlock,
        rows,
        cols,
        comm.rank(),
        comm.world_size(),
        local,
    )?;
    Ok((u, singular.column(0).to_vec(), vt))
}

/// Root-side half of [`distributed_svd`]: diagonalize `R` and pack the three
/// factors into one frame. `s` travels as an `n x 1` matrix so the whole
/// payload is a run of self-describing [`encode_matrix`] frames.
fn svd_of_r<T: DistFloat>(
    factorization: &TsqrFactorization<T>,
) -> Result<Vec<u8>, DistributedLinalgError> {
    let r = factorization.r();
    let (u_r, singular, vt) = jacobi_svd(&r)?;
    let n = r.nrows();
    if u_r.dim() != (n, n) || vt.dim() != (n, n) || singular.len() != n {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "SVD of an {n}x{n} R returned {:?} / {} / {:?}",
            u_r.dim(),
            singular.len(),
            vt.dim()
        )));
    }
    let singular_column = Array2::from_shape_vec((n, 1), singular.to_vec()).map_err(|e| {
        DistributedLinalgError::LinalgError(format!("cannot reshape the singular values: {e}"))
    })?;
    let mut payload = encode_matrix(&u_r.view());
    payload.extend_from_slice(&encode_matrix(&singular_column.view()));
    payload.extend_from_slice(&encode_matrix(&vt.view()));
    Ok(payload)
}

/// Least-squares solution of an overdetermined `A X = B`.
///
/// `A` is `m x n` row-block distributed, `B` is `m x k` under the same
/// distribution, and the returned `n x k` `X` is replicated. With `A = Q R`,
/// minimizing `||A X - B||` means solving `R X = Q^T B`, so this is one
/// upward pass of the tree followed by one triangular solve.
///
/// The solution is small (`n x k`) and every rank tends to want it, so it is
/// broadcast rather than left on the root.
///
/// # Errors
///
/// - [`DistributedLinalgError::UnsupportedShape`] unless *every* row block
///   is at least as tall as `A` is wide. That is stricter than the `m >= n`
///   a least-squares problem needs on its own: it is [`mod@super::tsqr`]'s leaf
///   precondition, and it bites when a modestly tall `A` is spread over many
///   ranks. See that module on CAQR, the algorithm for the regime this one
///   declines.
/// - [`DistributedLinalgError::SingularMatrix`] when `A` is rank deficient,
///   which leaves `R` with a negligible diagonal entry. This is a plain
///   triangular solve and offers no minimum-norm answer for that case; take
///   [`distributed_svd`] and truncate the small singular values instead.
pub async fn distributed_solve<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    b: &DistributedMatrix<T>,
    comm: &C,
) -> Result<Array2<T>, DistributedLinalgError> {
    check_rhs(a, b)?;
    let factorization = tsqr(a, comm).await?;
    let carried = factorization.apply_qt(comm, b.local_view()).await?;
    let ctx = comm.next_ctx();

    let produced = if comm.rank() == ROOT {
        match carried {
            Some(qt_b) => solve_upper_triangular(&factorization.r(), &qt_b.view())
                .map(|x| encode_matrix(&x.view())),
            None => Err(DistributedLinalgError::LinalgError(
                "the TSQR tree did not deliver Q^T B to its root".to_string(),
            )),
        }
    } else {
        Ok(Vec::new())
    };
    let x =
        decode_matrix::<T>(&bcast_fallible_bytes(comm, ROOT, ctx, TAG_SOLVE_X, produced).await?)?;

    let (_, cols) = a.global_shape();
    let (_, rhs_cols) = b.global_shape();
    if x.dim() != (cols, rhs_cols) {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "root broadcast a {:?} solution, expected {:?}",
            x.dim(),
            (cols, rhs_cols)
        )));
    }
    Ok(x)
}

/// Solve `A x = b` for a symmetric positive definite `A`, by Cholesky.
///
/// `A` must be in [`Layout::ColBlockCyclic`] — see [`super::cholesky`] for
/// why that layout and not row blocks. `b` and the returned solution are
/// replicated `n`-vectors.
pub async fn distributed_solve_spd<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    b: &[T],
    comm: &C,
) -> Result<Vec<T>, DistributedLinalgError> {
    super::cholesky::solve_spd(a, b, comm).await
}

/// Vector form of [`distributed_solve`]: `A x = b` for a single right-hand
/// side, returning the replicated `n`-vector.
pub async fn distributed_lstsq<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    b: &DistributedMatrix<T>,
    comm: &C,
) -> Result<Array1<T>, DistributedLinalgError> {
    let (_, rhs_cols) = b.global_shape();
    if rhs_cols != 1 {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "distributed_lstsq takes a single right-hand side, got {rhs_cols} columns"
        )));
    }
    let x = distributed_solve(a, b, comm).await?;
    Ok(x.column(0).to_owned())
}

/// Least-squares solve via QR — the spelling this module carried before the
/// [`DistributedMatrix`] surface existed. Delegates to
/// [`distributed_solve`].
pub async fn solve_via_qr<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    b: &DistributedMatrix<T>,
    comm: &C,
) -> Result<Array2<T>, DistributedLinalgError> {
    distributed_solve(a, b, comm).await
}

/// SVD via TSQR — the spelling this module carried before the
/// [`DistributedMatrix`] surface existed. Delegates to
/// [`distributed_svd`].
pub async fn svd_via_tsqr<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    comm: &C,
) -> Result<DistributedSvd<T>, DistributedLinalgError> {
    distributed_svd(a, comm).await
}

#[cfg(test)]
mod tests {
    use super::super::matrix::testutil::{deterministic_matrix, frobenius};
    use super::*;
    use crate::distributed::linalg::LocalFabric;
    use crate::distributed::testing::{LocalCluster, RankContext};
    use scirs2_core::ndarray::s;
    use std::sync::Arc;

    fn sorted_descending(values: &[f64]) -> Vec<f64> {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    fn eye(n: usize) -> Array2<f64> {
        identity::<f64>(n)
    }

    /// Rebuild `U diag(s) V^T` and measure the distance back to `a`.
    fn reconstruction_error(u: &Array2<f64>, s: &[f64], vt: &Array2<f64>, a: &Array2<f64>) -> f64 {
        let mut scaled = u.clone();
        for (j, value) in s.iter().enumerate() {
            let mut column = scaled.column_mut(j);
            column *= *value;
        }
        frobenius(&(&scaled.dot(vt) - a).view())
    }

    #[test]
    fn jacobi_svd_reconstructs_and_is_orthonormal() {
        for (m, n) in [(1usize, 1usize), (4, 4), (9, 3), (6, 6), (12, 5)] {
            let a = deterministic_matrix(m, n, 31 + m as u64);
            let (u, s, vt) = jacobi_svd(&a.view()).expect("jacobi svd");
            assert_eq!(u.dim(), (m, n));
            assert_eq!(vt.dim(), (n, n));
            assert_eq!(s.len(), n);

            let error = reconstruction_error(&u, &s, &vt, &a);
            assert!(error < 1e-13, "{m}x{n}: ||U S V^T - A||_F = {error}");

            let u_ortho = &u.t().dot(&u) - &eye(n);
            assert!(
                frobenius(&u_ortho.view()) < 1e-13,
                "{m}x{n}: U not orthonormal"
            );
            let v_ortho = &vt.dot(&vt.t()) - &eye(n);
            assert!(
                frobenius(&v_ortho.view()) < 1e-13,
                "{m}x{n}: V not orthonormal"
            );
            assert_eq!(sorted_descending(&s), s, "{m}x{n}: values must be ordered");
        }
    }

    /// A rank-deficient input says nothing about the trailing columns of
    /// `U`; they must still come back as a genuine orthonormal completion,
    /// not as zeros that would quietly break `U^T U = I` downstream.
    #[test]
    fn jacobi_svd_completes_the_basis_when_rank_deficient() {
        let base = deterministic_matrix(7, 2, 404);
        let mut a = Array2::<f64>::zeros((7, 4));
        a.slice_mut(s![.., ..2]).assign(&base);
        // Columns 2 and 3 are copies of column 0: rank 2 out of 4.
        for i in 0..7 {
            a[[i, 2]] = base[[i, 0]];
            a[[i, 3]] = base[[i, 0]];
        }

        let (u, s, vt) = jacobi_svd(&a.view()).expect("jacobi svd");
        assert!(s[2] < 1e-14 && s[3] < 1e-14, "expected rank 2, got {s:?}");
        let error = reconstruction_error(&u, &s, &vt, &a);
        assert!(error < 1e-13, "||U S V^T - A||_F = {error}");
        let u_ortho = &u.t().dot(&u) - &eye(4);
        assert!(
            frobenius(&u_ortho.view()) < 1e-13,
            "U not orthonormal: {}",
            frobenius(&u_ortho.view())
        );
    }

    /// The reason this module does not call out for its small SVD: a graded
    /// matrix whose singular values span twelve orders of magnitude. Forming
    /// `A^T A` pushes the smallest eigenvalue to ~1e-24, far under `eps`
    /// relative to the largest, and the corresponding singular value is
    /// simply lost. One-sided Jacobi never forms that product and resolves
    /// it to full relative accuracy.
    #[test]
    fn jacobi_svd_keeps_small_singular_values_a_cross_product_method_loses() {
        // Exactly representable, well conditioned (singular values ~0.38 to
        // ~1.87), so the scaled columns pin each singular value's magnitude.
        let well_conditioned =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 1.0])
                .expect("3x3");
        let scales = [1.0_f64, 1e-6, 1e-12];
        let mut a = well_conditioned.clone();
        for (j, scale) in scales.iter().enumerate() {
            let mut column = a.column_mut(j);
            column *= *scale;
        }

        let (u, s, vt) = jacobi_svd(&a.view()).expect("jacobi svd");
        assert!(
            s[2] > 1e-13 && s[2] < 1e-11,
            "smallest singular value {} is outside the band the scaling dictates",
            s[2]
        );
        let error = reconstruction_error(&u, &s, &vt, &a);
        assert!(error < 1e-15, "||U S V^T - A||_F = {error}");
        let u_ortho = &u.t().dot(&u) - &eye(3);
        assert!(frobenius(&u_ortho.view()) < 1e-13);
    }

    /// The kernel is generic over [`DistFloat`], and `f32` is the other
    /// instantiation the distributed entry points can be called with.
    #[test]
    fn jacobi_svd_handles_f32() {
        let a = deterministic_matrix(9, 4, 313).mapv(|v| v as f32);
        let (u, s, vt) = jacobi_svd(&a.view()).expect("jacobi svd");
        let mut scaled = u.clone();
        for (j, value) in s.iter().enumerate() {
            let mut column = scaled.column_mut(j);
            column *= *value;
        }
        let error = frobenius(&(&scaled.dot(&vt) - &a).mapv(f64::from).view());
        assert!(error < 1e-5, "f32: ||U S V^T - A||_F = {error}");
        let ortho = (&u.t().dot(&u) - &identity::<f32>(4)).mapv(f64::from);
        assert!(frobenius(&ortho.view()) < 1e-5, "f32: U not orthonormal");
    }

    /// Extreme magnitudes must not defeat the sweep, in either direction.
    ///
    /// The sweep works with sums of squares, which is a doubly quadratic
    /// quantity and so runs out of exponent range long before the data does.
    /// Two distinct failures lurk, and both are *silent*:
    ///
    /// - comparing `|gamma|` against `sqrt(alpha * beta)` squares the norms
    ///   a second time, overflowing at an `f32` column norm of only ~6e9;
    /// - `alpha` itself overflows once entries reach ~1e19 in `f32`, and
    ///   underflows to zero once they fall to ~1e-23.
    ///
    /// In every case the pair reads as "already orthogonal", the sweep ends
    /// at once, and the routine returns the normalized input columns with
    /// `V = I` — which reconstructs `A` perfectly while `U` is not
    /// orthonormal at all. Only the orthonormality assertion catches that;
    /// a reconstruction check alone passes happily.
    #[test]
    fn jacobi_svd_survives_columns_near_the_overflow_edge() {
        for scale in [1e-30_f32, 1e-9, 1e9, 1e14, 1e18, 1e30] {
            let a = deterministic_matrix(6, 3, 191).mapv(|v| v as f32 * scale);
            let (u, s, vt) = jacobi_svd(&a.view()).expect("jacobi svd");

            let ortho = (&u.t().dot(&u) - &identity::<f32>(3)).mapv(f64::from);
            assert!(
                frobenius(&ortho.view()) < 1e-4,
                "scale={scale:e}: U not orthonormal, ||U^T U - I||_F = {}",
                frobenius(&ortho.view())
            );
            assert!(
                s.iter().all(|value| value.is_finite()),
                "scale={scale:e}: non-finite singular values {s:?}"
            );

            let mut scaled = u.clone();
            for (j, value) in s.iter().enumerate() {
                let mut column = scaled.column_mut(j);
                column *= *value;
            }
            let relative = frobenius(&(&scaled.dot(&vt) - &a).mapv(f64::from).view())
                / frobenius(&a.mapv(f64::from).view());
            assert!(
                relative < 1e-5,
                "scale={scale:e}: relative error {relative}"
            );
        }
    }

    #[test]
    fn jacobi_svd_refuses_a_wide_matrix() {
        let a = deterministic_matrix(2, 5, 1);
        assert!(matches!(
            jacobi_svd(&a.view()),
            Err(DistributedLinalgError::UnsupportedShape(_))
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distributed_qr_reconstructs_and_replicates_r() {
        for world_size in 1..=4u32 {
            let a = deterministic_matrix(32, 4, 1234);
            let fabric = LocalFabric::new(world_size);
            let reference = a.clone();
            let results = LocalCluster::run(world_size, move |ctx: RankContext| {
                let fabric = Arc::clone(&fabric);
                let a = reference.clone();
                async move {
                    let comm = fabric.transport(ctx.rank)?;
                    let da = DistributedMatrix::from_global(
                        Layout::RowBlock,
                        &a.view(),
                        ctx.rank,
                        ctx.world_size,
                    )?;
                    let (q, r) = distributed_qr(&da, &comm).await?;
                    Ok((q.gather_to_root(&comm, 0).await?, r))
                }
            })
            .await
            .expect("cluster run should succeed");

            let (gathered, r) = results.first().cloned().expect("root result");
            let q = gathered.expect("root gathers Q");
            let diff = &q.dot(&r) - &a;
            assert!(
                frobenius(&diff.view()) < 1e-10,
                "p={world_size}: ||QR - A||_F = {}",
                frobenius(&diff.view())
            );
            for (rank, (_, other)) in results.iter().enumerate() {
                assert_eq!(other, &r, "rank {rank} disagrees about R");
            }
        }
    }

    /// The reconstruction and orthonormality checks are reference-free: if
    /// `A = U diag(s) V^T` with orthonormal `U` and `V` and non-negative
    /// decreasing `s`, then `s` *are* the singular values of `A`. The
    /// comparison against the local solver on the gathered matrix then
    /// confirms the two agree.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distributed_svd_matches_the_local_decomposition() {
        for world_size in 1..=4u32 {
            let a = deterministic_matrix(32, 4, 555);
            let fabric = LocalFabric::new(world_size);
            let reference = a.clone();
            let results = LocalCluster::run(world_size, move |ctx: RankContext| {
                let fabric = Arc::clone(&fabric);
                let a = reference.clone();
                async move {
                    let comm = fabric.transport(ctx.rank)?;
                    let da = DistributedMatrix::from_global(
                        Layout::RowBlock,
                        &a.view(),
                        ctx.rank,
                        ctx.world_size,
                    )?;
                    let (u, s, vt) = distributed_svd(&da, &comm).await?;
                    Ok((u.gather_to_root(&comm, 0).await?, s, vt))
                }
            })
            .await
            .expect("cluster run should succeed");

            let (gathered, s, vt) = results.first().cloned().expect("root result");
            let u = gathered.expect("root gathers U");

            let error = reconstruction_error(&u, &s, &vt, &a);
            assert!(error < 1e-10, "p={world_size}: ||U S V^T - A||_F = {error}");

            let u_ortho = &u.t().dot(&u) - &eye(4);
            assert!(
                frobenius(&u_ortho.view()) < 1e-10,
                "p={world_size}: U not orthonormal"
            );
            let v_ortho = &vt.dot(&vt.t()) - &eye(4);
            assert!(
                frobenius(&v_ortho.view()) < 1e-10,
                "p={world_size}: V not orthonormal"
            );

            assert_eq!(sorted_descending(&s), s, "singular values must be ordered");
            assert!(s.iter().all(|v| *v >= 0.0));

            // Agreement with the local solver on the gathered matrix. The
            // tolerance is loose *on purpose*: `scirs2_linalg::svd` reaches
            // its answer through the eigendecomposition of `A^T A`, which
            // squares the condition number and so resolves the spectrum to
            // about `sqrt(eps)` rather than `eps` — its own factorization of
            // this very matrix misses by ~6e-4 in Frobenius norm, which is
            // why `jacobi_svd` exists. The strict assertions above are what
            // actually pin this result down; this one only confirms the two
            // methods describe the same matrix.
            let (_, local_s, _) = scirs2_linalg::svd(&a.view(), false, None).expect("local svd");
            let expected = sorted_descending(local_s.as_slice().unwrap_or(&[]));
            assert_eq!(expected.len(), s.len());
            let scale = s.first().copied().unwrap_or(1.0).max(1.0);
            for (got, want) in s.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < 1e-6 * scale,
                    "p={world_size}: singular value {got} vs local {want}"
                );
            }

            for (rank, (_, other_s, other_vt)) in results.iter().enumerate() {
                assert_eq!(other_s, &s, "rank {rank} disagrees about s");
                assert_eq!(other_vt, &vt, "rank {rank} disagrees about V^T");
            }
        }
    }

    /// The defining property of a least-squares solution is that the
    /// residual is orthogonal to the column space: `A^T (A x - b) = 0`. That
    /// is checked first because it needs no reference implementation at all;
    /// the comparison against the local solver follows.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distributed_solve_satisfies_the_normal_equations() {
        for world_size in 1..=4u32 {
            let a = deterministic_matrix(32, 4, 3131);
            let b = deterministic_matrix(32, 1, 4141);
            let fabric = LocalFabric::new(world_size);
            let (a_ref, b_ref) = (a.clone(), b.clone());
            let results = LocalCluster::run(world_size, move |ctx: RankContext| {
                let fabric = Arc::clone(&fabric);
                let (a, b) = (a_ref.clone(), b_ref.clone());
                async move {
                    let comm = fabric.transport(ctx.rank)?;
                    let da = DistributedMatrix::from_global(
                        Layout::RowBlock,
                        &a.view(),
                        ctx.rank,
                        ctx.world_size,
                    )?;
                    let db = DistributedMatrix::from_global(
                        Layout::RowBlock,
                        &b.view(),
                        ctx.rank,
                        ctx.world_size,
                    )?;
                    Ok(distributed_solve(&da, &db, &comm).await?)
                }
            })
            .await
            .expect("cluster run should succeed");

            let x = results.first().cloned().expect("root result");
            assert_eq!(x.dim(), (4, 1));

            let residual = &a.dot(&x) - &b;
            let normal = a.t().dot(&residual);
            let scale = frobenius(&a.t().dot(&b).view()).max(1.0);
            assert!(
                frobenius(&normal.view()) < 1e-10 * scale,
                "p={world_size}: ||A^T (A x - b)||_F = {}",
                frobenius(&normal.view())
            );

            let expected =
                scirs2_linalg::lstsq(&a.view(), &b.column(0), None).expect("local lstsq");
            for i in 0..4 {
                assert!(
                    (x[[i, 0]] - expected.x[i]).abs() < 1e-9,
                    "p={world_size}: x[{i}] = {} vs local {}",
                    x[[i, 0]],
                    expected.x[i]
                );
            }

            for (rank, other) in results.iter().enumerate() {
                assert_eq!(other, &x, "rank {rank} disagrees about the solution");
            }
        }
    }

    /// Several right-hand sides at once must give the same answer as solving
    /// them one at a time.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distributed_solve_handles_multiple_right_hand_sides() {
        let a = deterministic_matrix(24, 3, 707);
        let b = deterministic_matrix(24, 3, 808);
        let fabric = LocalFabric::new(3);
        let (a_ref, b_ref) = (a.clone(), b.clone());
        let results = LocalCluster::run(3, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            let (a, b) = (a_ref.clone(), b_ref.clone());
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let da = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let db = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &b.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let all = distributed_solve(&da, &db, &comm).await?;
                let first = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &b.slice(s![.., ..1]),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let single = distributed_lstsq(&da, &first, &comm).await?;
                Ok((all, single))
            }
        })
        .await
        .expect("cluster run should succeed");

        let (all, single) = results.first().cloned().expect("root result");
        assert_eq!(all.dim(), (3, 3));
        for i in 0..3 {
            assert!((all[[i, 0]] - single[i]).abs() < 1e-12);
        }
        let residual = &a.dot(&all) - &b;
        let normal = a.t().dot(&residual);
        assert!(frobenius(&normal.view()) < 1e-10 * frobenius(&a.t().dot(&b).view()).max(1.0));
    }

    /// The Cholesky path is reachable through this module's own entry point,
    /// and it wants the other layout.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distributed_solve_spd_delegates_to_cholesky() {
        use super::super::matrix::testutil::spd_matrix;

        let n = 24usize;
        let a = spd_matrix(n, 9090);
        let x_true: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
        let b: Vec<f64> = a
            .rows()
            .into_iter()
            .map(|row| row.iter().zip(x_true.iter()).map(|(v, x)| v * x).sum())
            .collect();

        let fabric = LocalFabric::new(4);
        let (a_ref, b_ref) = (a.clone(), b.clone());
        let results = LocalCluster::run(4, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            let (a, b) = (a_ref.clone(), b_ref.clone());
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let da = DistributedMatrix::from_global(
                    Layout::ColBlockCyclic { panel_width: 5 },
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                Ok(distributed_solve_spd(&da, &b, &comm).await?)
            }
        })
        .await
        .expect("cluster run should succeed");

        for (rank, x) in results.iter().enumerate() {
            for (i, (got, want)) in x.iter().zip(x_true.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-9,
                    "rank {rank}: x[{i}] = {got} vs {want}"
                );
            }
        }
    }

    /// A rank-deficient `A` must surface as [`DistributedLinalgError::SingularMatrix`]
    /// on *every* rank, not just the root that ran the triangular solve.
    /// Only the root can discover it, so the verdict has to survive the
    /// broadcast framing with its variant intact — a generic message would
    /// leave `matches!(err, SingularMatrix)` true on one rank and false on
    /// the rest, which is worse than either answer alone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_rank_deficient_system_reports_singular_on_every_rank() {
        let base = deterministic_matrix(20, 3, 616);
        let mut a = Array2::<f64>::zeros((20, 4));
        a.slice_mut(s![.., ..3]).assign(&base);
        // The fourth column repeats the first: R's last diagonal entry
        // collapses and the back substitution has nothing to divide by.
        for i in 0..20 {
            a[[i, 3]] = base[[i, 0]];
        }
        let b = deterministic_matrix(20, 1, 626);

        let fabric = LocalFabric::new(4);
        let results = LocalCluster::run(4, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            let (a, b) = (a.clone(), b.clone());
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let da = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let db = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &b.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                Ok(matches!(
                    distributed_solve(&da, &db, &comm).await,
                    Err(DistributedLinalgError::SingularMatrix)
                ))
            }
        })
        .await
        .expect("cluster run should succeed");

        for (rank, singular) in results.iter().enumerate() {
            assert!(*singular, "rank {rank} did not report SingularMatrix");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_mismatched_right_hand_side_is_refused() {
        let fabric = LocalFabric::new(1);
        let results = LocalCluster::run(1, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let a = deterministic_matrix(12, 3, 1);
                let b = deterministic_matrix(11, 1, 2);
                let da = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let db = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &b.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                Ok(matches!(
                    distributed_solve(&da, &db, &comm).await,
                    Err(DistributedLinalgError::DimensionMismatch(_))
                ))
            }
        })
        .await
        .expect("cluster run should succeed");
        assert_eq!(results.first(), Some(&true));
    }
}
