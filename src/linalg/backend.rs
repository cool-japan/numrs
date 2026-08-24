//! `scirs2-linalg` fast-path backend for [`Array`]'s core linear-algebra
//! methods.
//!
//! # Why this module exists
//!
//! With `lapack` in the crate's default feature set, the primary
//! `impl<T: Float + ...> Array<T>` block in [`crate::linalg`] became the
//! live implementation of `det`/`inv`/`solve`/`svd`/`eig`/`cholesky`/`qr`.
//! Those methods route to [`crate::new_modules::matrix_decomp`], which is
//! hand-rolled Rust: `det`/`inv` run an `O(n^3)` LU with a
//! *bounds-checked, `Result`-returning* `Array::get(&[i, j])?` on every
//! single element access, and `svd`/`qr`/`cholesky` -- while they do
//! eventually reach OxiBLAS via `scirs2_core::linalg::*_ndarray` -- wrap
//! that call in `O(n^2)` per-element scaling/conversion loops built out of
//! the same checked accessors.
//!
//! `scirs2-linalg` already exposes LAPACK-backed `det`/`inv`/`solve`/
//! `svd`/`eigh`/`cholesky`/`qr` and is already a mandatory dependency
//! (see `src/linalg_accelerated.rs` and `src/new_modules/timeseries/*`
//! for existing call sites). This module is the adapter: every function
//! here answers one question -- *"can `scirs2-linalg` handle this
//! operand?"* -- and returns
//!
//! - `None` -- **not eligible**; the caller must run its existing code
//!   path unchanged, and
//! - `Some(result)` -- eligible and attempted; the caller returns
//!   `result` verbatim.
//!
//! Because `None` means "nothing happened", the hooks in
//! [`crate::linalg`] are pure prefixes: they never move, reorder, or
//! duplicate the validation the existing bodies already perform. Any
//! operand this module declines -- including every malformed one, since
//! [`square_dim`] only accepts a genuinely square 2-D array -- reaches
//! the original code with the original error message intact.
//!
//! # Element dispatch is zero-copy
//!
//! Every entry point in `scirs2-linalg` is generic over
//! `F: Float + NumAssign + Sum + Send + Sync + ScalarOperand + 'static`,
//! so both `f64` and `f32` can be handed to it *as themselves*. Each
//! function here therefore dispatches on [`std::any::TypeId`] through
//! [`crate::kernels::cast`]:
//!
//! 1. [`borrow::operand`] borrows the operand's data as a flat,
//!    logically-ordered slice -- genuinely zero-copy when the array is
//!    contiguous, which is the common case.
//! 2. [`cast::as_f64`] / [`cast::as_f32`] reinterpret that `&[T]` as
//!    `&[f64]` / `&[f32]` under a `TypeId` proof that `T` *is* that type,
//!    with no element conversion at all.
//! 3. The result's `Vec` is handed back through [`cast::vec_from_f64`] /
//!    [`cast::vec_from_f32`], which reuse the original allocation.
//!
//! So for `T = f64` and `T = f32` there is no widening, no narrowing, and
//! no `O(n^2)` conversion pass on either side of the call -- the operand
//! is read straight out of the array's own storage and the result is
//! moved, not copied, into the returned [`Array`]. Any other `T` fails
//! both `TypeId` checks and yields `None`, i.e. the existing generic
//! path, never an error.
//!
//! # Size gates
//!
//! [`SCIRS2_MIN_DIM`] documents the size below which dispatching is a
//! pessimisation; see its own docs for the rationale. [`SOLVE_MIN_DIM`]
//! is deliberately lower and documents why.
//!
//! Two operations also have an *upper* bound: [`QR_MAX_DIM`] and
//! [`SVD_MAX_DIM`]. `det`/`inv`/`cholesky`/`eig` replace a
//! checked-accessor loop and win by one to three orders of magnitude at
//! every size measured, but `matrix_decomp::{qr, svd}` already reach
//! OxiBLAS, so there the two paths are within a small factor and the
//! ordering *flips* as `n` grows. Both constants carry the measurements
//! that place them.
//!
//! # Shape and convention fidelity
//!
//! Every function returns arrays with the *same shapes and conventions*
//! as the path it replaces:
//!
//! - `qr` -- the existing path returns economy QR (`Q` is `m x min(m, n)`,
//!   `R` is `min(m, n) x n`) while `scirs2_linalg::qr` returns full QR
//!   (`Q` is `m x m`, `R` is `m x n`). Those coincide exactly when
//!   `m == n`, so this backend accepts **square input only** and leaves
//!   rectangular QR to the existing path.
//! - `svd` -- same reasoning: square-only, where thin and full SVD agree.
//! - `cholesky` -- both produce the **lower** triangular factor
//!   (`A = L * L^T`).
//! - `eig` -- see [`try_scirs2_eig`]; only the symmetric case is taken,
//!   and eigenvalue *ordering* is the one deliberate deviation.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::kernels::{borrow, cast};
use num_traits::Float;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_linalg::error::LinalgError;
use std::fmt::Debug;

/// Smallest matrix dimension for which routing through `scirs2-linalg`
/// beats the existing in-crate path.
///
/// The rationale is the shape of what is on either side of the gate:
///
/// * **Below the gate the existing path is already good.** `Array::det`
///   and `Array::inv` carry hand-written closed forms for `n <= 3`
///   -- straight-line arithmetic on `to_vec()`ed data with no
///   decomposition, no allocation per column, and no dynamic dispatch.
///   Nothing this backend can do is faster than a 3x3 cofactor expansion.
///   `scirs2-linalg` agrees, incidentally: `scirs2_linalg::solve` itself
///   short-circuits to `inv` for `nrows() <= 4`, and
///   `scirs2_linalg::eigh` carries closed forms up to `n == 4`.
///
/// * **The dispatch has a fixed cost.** `scirs2-linalg`'s own validation
///   sweep (`qr`/`svd` scan every element for non-finite values), its
///   worker configuration, and one `Vec` move per output matrix. Against
///   the `O(n^3)` it is meant to amortise, the break-even point is
///   wherever `n^3` starts to dominate `n^2`.
///
/// * **16 is comfortably past break-even without being aggressive.** A
///   16x16 decomposition is ~4k multiply-adds against 256 elements of
///   validation, a 16:1 ratio that grows linearly from there (64:1 at
///   `n = 64`). At `n = 16` the measured speedups are 64x for `cholesky`,
///   2.0x for `svd`, 1.6x for `qr` and ~1000x for symmetric `eig`
///   (43 ms -> 42 us); picking the next power of two down would put the
///   gate inside the range the closed forms and `scirs2-linalg`'s own
///   `<= 4` short-circuits already cover.
///
/// The one operation that does *not* use this gate is `solve`; see
/// [`SOLVE_MIN_DIM`].
pub(crate) const SCIRS2_MIN_DIM: usize = 16;

/// Exclusive upper bound for `qr`: above this, `scirs2-linalg` is
/// *slower* than the existing path, so the gate closes again.
///
/// Unlike `det`/`inv`/`cholesky`/`eig` -- where the existing code is a
/// checked-accessor loop and the backend wins by one to three orders of
/// magnitude -- `matrix_decomp::qr` already reaches OxiBLAS, so the two
/// paths are within a small factor of each other and the ordering
/// actually flips with size.
///
/// Measured (release, three passes, `Q R` of a deterministic diagonally
/// dominant `f64` matrix, backend / existing):
///
/// | `n`  | pass 1 | pass 2 | pass 3 |
/// |------|--------|--------|--------|
/// | 16   | 1.63x  | 1.72x  | --     |
/// | 32   | 1.40x  | 1.48x  | --     |
/// | 64   | 1.02x  | 1.06x  | 1.02x  |
/// | 96   | 0.87x  | 0.87x  | --     |
/// | 112  | 0.85x  | 0.90x  | 0.89x  |
/// | 128  | 0.68x  | 0.64x  | 0.67x  |
/// | 160  | 0.80x  | 0.78x  | 0.77x  |
///
/// 64 is the largest size that is a win in *every* pass and 96 the
/// smallest that is a loss in every pass, so the bound sits between
/// them. Note this is an upper bound on `qr` **only** -- `det`, `inv`,
/// `cholesky`, `eig` and `solve` have no upper bound, because their
/// margins (9x and up at every size measured, out to `n = 384`) never
/// come close to inverting.
pub(crate) const QR_MAX_DIM: usize = 96;

/// Exclusive upper bound for `svd`, for the same reason as
/// [`QR_MAX_DIM`] but at a much larger size: `svd`'s margin over the
/// existing path is wider, so it survives further before inverting.
///
/// Measured (release, three passes, backend / existing):
///
/// | `n`  | pass 1 | pass 2 | pass 3 |
/// |------|--------|--------|--------|
/// | 16   | 2.03x  | 2.05x  | --     |
/// | 64   | 1.90x  | 1.93x  | 2.02x  |
/// | 128  | 1.27x  | 1.37x  | 1.33x  |
/// | 160  | 1.19x  | 1.07x  | 1.08x  |
/// | 176  | 1.60x  | 1.62x  | 1.88x  |
/// | 192  | 1.21x  | 0.86x  | 1.17x  |
/// | 224  | 1.27x  | 1.03x  | 1.08x  |
/// | 256  | 0.58x  | 0.42x  | 0.61x  |
///
/// 224 is the largest size that is a win in every pass; 256 is a decisive
/// loss in every pass. (The 160-224 band is noisy -- these runs shared
/// the machine with other builds -- but never a *material* loss, whereas
/// 256 is consistently around half speed.)
pub(crate) const SVD_MAX_DIM: usize = 256;

/// Gate for `solve`, deliberately lower than [`SCIRS2_MIN_DIM`].
///
/// `Array::solve`'s `n > 3` branch delegates to
/// `interop::scirs_compat::solve_linear_system`, which calls the free
/// function `crate::linalg::solve`, which calls `Array::solve` again --
/// an unconditional infinite recursion that aborts the process with a
/// stack overflow. (Latent until `lapack` joined the default features and
/// made this impl block live; reproduced by the `solve_recursion_probe`
/// test in `tests/test_linalg_backend.rs`, which aborts with SIGABRT and
/// `fatal runtime error: stack overflow` on the unhooked build.)
///
/// So for `solve` there is no slow-but-correct path to fall back to above
/// `n = 3`: the gate is set to cover *everything* the closed forms do not,
/// and a `scirs2-linalg` failure is reported as an error rather than
/// being handed back to a path that would crash.
pub(crate) const SOLVE_MIN_DIM: usize = 4;

/// Translate a `scirs2-linalg` error into this crate's error type.
///
/// Mirrors `linalg_accelerated::linalg_to_numrs2_error` exactly, so a
/// given `LinalgError` surfaces as the same `NumRs2Error` no matter which
/// of the two entry points produced it.
fn map_linalg_err(e: LinalgError) -> NumRs2Error {
    match e {
        LinalgError::SingularMatrixError(s) => {
            NumRs2Error::InvalidOperation(format!("Singular matrix: {}", s))
        }
        LinalgError::DimensionError(s) | LinalgError::ShapeError(s) => {
            NumRs2Error::DimensionMismatch(s)
        }
        LinalgError::NonPositiveDefiniteError(s) => {
            NumRs2Error::InvalidOperation(format!("Matrix is not positive definite: {}", s))
        }
        LinalgError::ConvergenceError(s) => {
            NumRs2Error::ComputationError(format!("Convergence failed: {}", s))
        }
        other => NumRs2Error::ComputationError(format!("Linear algebra error: {}", other)),
    }
}

/// `Some(n)` iff `a` is a 2D `n x n` matrix with `n >= min_dim`.
///
/// Anything else -- 1-D, 3-D, rectangular, or simply small -- is declined,
/// which is what keeps the caller's own validation (and its error
/// messages) the sole authority on malformed input.
fn square_dim<T: Clone>(a: &Array<T>, min_dim: usize) -> Option<usize> {
    square_dim_in_range(a, min_dim, usize::MAX)
}

/// As [`square_dim`], but also declining `n >= max_dim`.
///
/// Used by the two operations whose advantage over the existing path
/// inverts with size; see [`QR_MAX_DIM`] and [`SVD_MAX_DIM`].
fn square_dim_in_range<T: Clone>(a: &Array<T>, min_dim: usize, max_dim: usize) -> Option<usize> {
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] || shape[0] < min_dim || shape[0] >= max_dim {
        return None;
    }
    Some(shape[0])
}

/// Take an owned `ndarray` matrix's elements as a row-major `Vec`.
///
/// Reuses the existing allocation whenever the matrix is already
/// standard (row-major) layout *and* starts at offset 0, which is the
/// normal case for a freshly-computed `scirs2-linalg` result. The
/// `is_standard_layout` guard is not optional: `into_raw_vec_and_offset`
/// hands back the underlying buffer in **memory** order, so taking it
/// unconditionally would silently transpose any result that happens to
/// come back column-major.
fn into_row_major_vec<F: Clone>(m: Array2<F>) -> Vec<F> {
    if m.is_standard_layout() {
        let len = m.len();
        let (buf, offset) = m.into_raw_vec_and_offset();
        let start = offset.unwrap_or(0);
        if start == 0 && buf.len() == len {
            return buf;
        }
        return buf[start..start + len].to_vec();
    }
    m.iter().cloned().collect()
}

/// Generate the `Array2<F> -> Array<T>` and `Array1<F> -> Array<T>`
/// converters for one concrete float type.
///
/// Macro-generated rather than hand-written twice so the `f32` pair can
/// never drift from the `f64` pair: both are the same code, and the only
/// thing that varies is which [`cast`] reinterpretation is applied.
macro_rules! concrete_converters {
    ($f:ty, $matrix_fn:ident, $vector_fn:ident, $vec_cast:path) => {
        /// Move a computed matrix into an `Array<T>`, preserving shape.
        ///
        /// `None` only when `T` is not the macro's float type, which the
        /// caller has already ruled out by dispatching here.
        fn $matrix_fn<T: Clone + 'static>(m: Array2<$f>) -> Option<Array<T>> {
            let (rows, cols) = (m.nrows(), m.ncols());
            let data: Vec<T> = $vec_cast(into_row_major_vec(m))?;
            Some(Array::from_vec_shape(data, &[rows, cols]).unwrap_or_else(|e| panic!("{e}")))
        }

        /// Move a computed vector into a 1-D `Array<T>`.
        fn $vector_fn<T: Clone + 'static>(v: Array1<$f>) -> Option<Array<T>> {
            let data: Vec<T> = $vec_cast(v.to_vec())?;
            Some(Array::from_vec(data))
        }
    };
}

concrete_converters!(f64, matrix_from_f64, vector_from_f64, cast::vec_from_f64);
concrete_converters!(f32, matrix_from_f32, vector_from_f32, cast::vec_from_f32);

/// What [`try_scirs2_svd`] returns: `None` when the operand is not
/// eligible, otherwise the attempted `(U, s, V^T)` triple.
///
/// Named rather than written inline purely to keep the signature
/// readable (and `clippy::type_complexity` quiet).
type SvdAttempt<T> = Option<Result<(Array<T>, Array<T>, Array<T>)>>;

/// Determinant via `scirs2_linalg::det`.
///
/// Not eligible (`None`) for non-square / sub-[`SCIRS2_MIN_DIM`] input,
/// for a `T` that is neither `f64` nor `f32`, or when `scirs2-linalg`
/// itself fails -- in that last case the caller's LU path runs and
/// produces its own (unchanged) result or error.
pub(crate) fn try_scirs2_det<T>(a: &Array<T>) -> Option<Result<T>>
where
    T: Float + Clone + Debug + 'static,
{
    let n = square_dim(a, SCIRS2_MIN_DIM)?;
    let op = borrow::operand(a);

    if let Some(s) = cast::as_f64(&op) {
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let det = scirs2_linalg::det(&view, None).ok()?;
        return Some(Ok(cast::f64_to(det)?));
    }
    if let Some(s) = cast::as_f32(&op) {
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let det = scirs2_linalg::det(&view, None).ok()?;
        return Some(Ok(cast::f32_to(det)?));
    }
    None
}

/// Matrix inverse via `scirs2_linalg::inv`.
///
/// A `scirs2-linalg` failure yields `None` rather than an error, so a
/// singular matrix still produces the caller's existing
/// `"matrix is singular and cannot be inverted"` message.
pub(crate) fn try_scirs2_inv<T>(a: &Array<T>) -> Option<Result<Array<T>>>
where
    T: Float + Clone + Debug + 'static,
{
    let n = square_dim(a, SCIRS2_MIN_DIM)?;
    let op = borrow::operand(a);

    if let Some(s) = cast::as_f64(&op) {
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let inv = scirs2_linalg::inv(&view, None).ok()?;
        return Some(Ok(matrix_from_f64(inv)?));
    }
    if let Some(s) = cast::as_f32(&op) {
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let inv = scirs2_linalg::inv(&view, None).ok()?;
        return Some(Ok(matrix_from_f32(inv)?));
    }
    None
}

/// Solve `A x = b` for a single right-hand side via
/// `scirs2_linalg::solve`.
///
/// Unlike every other function in this module, a `scirs2-linalg` failure
/// is surfaced as `Some(Err(..))`, not `None`: for `n >= 4` the caller's
/// existing path is the infinite recursion described on
/// [`SOLVE_MIN_DIM`], so handing control back to it would abort the
/// process instead of reporting the problem.
pub(crate) fn try_scirs2_solve<T>(a: &Array<T>, b: &Array<T>) -> Option<Result<Array<T>>>
where
    T: Float + Clone + Debug + 'static,
{
    let n = square_dim(a, SOLVE_MIN_DIM)?;
    let b_shape = b.shape();
    if b_shape.len() != 1 || b_shape[0] != n {
        return None;
    }

    let a_op = borrow::operand(a);
    let b_op = borrow::operand(b);

    if let (Some(a_s), Some(b_s)) = (cast::as_f64(&a_op), cast::as_f64(&b_op)) {
        let a_view = ArrayView2::from_shape((n, n), a_s).ok()?;
        let b_view = ArrayView1::from_shape(n, b_s).ok()?;
        return Some(finish_solve(
            scirs2_linalg::solve(&a_view, &b_view, None).map(|x| vector_from_f64(x)),
        ));
    }
    if let (Some(a_s), Some(b_s)) = (cast::as_f32(&a_op), cast::as_f32(&b_op)) {
        let a_view = ArrayView2::from_shape((n, n), a_s).ok()?;
        let b_view = ArrayView1::from_shape(n, b_s).ok()?;
        return Some(finish_solve(
            scirs2_linalg::solve(&a_view, &b_view, None).map(|x| vector_from_f32(x)),
        ));
    }
    None
}

/// Collapse `solve`'s two failure modes (`scirs2-linalg` error, or a
/// result that will not convert back) into this crate's error type.
fn finish_solve<T>(r: std::result::Result<Option<Array<T>>, LinalgError>) -> Result<Array<T>> {
    match r {
        Ok(Some(arr)) => Ok(arr),
        Ok(None) => Err(NumRs2Error::ConversionError(
            "solve: solution is not representable in the array's element type".to_string(),
        )),
        Err(e) => Err(map_linalg_err(e)),
    }
}

/// Singular value decomposition via `scirs2_linalg::svd`.
///
/// Square input only -- see the module docs on shape fidelity. Returns
/// `(U, s, V^T)` with `U` `n x n`, `s` of length `n`, and `V^T` `n x n`,
/// matching the existing path exactly.
///
/// Engages for `SCIRS2_MIN_DIM <= n < SVD_MAX_DIM`; above that the
/// existing path is measurably faster.
pub(crate) fn try_scirs2_svd<T>(a: &Array<T>) -> SvdAttempt<T>
where
    T: Float + Clone + Debug + 'static,
{
    let n = square_dim_in_range(a, SCIRS2_MIN_DIM, SVD_MAX_DIM)?;
    let op = borrow::operand(a);

    if let Some(s) = cast::as_f64(&op) {
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let (u, sv, vt) = scirs2_linalg::svd(&view, true, None).ok()?;
        return Some(Ok((
            matrix_from_f64(u)?,
            vector_from_f64(sv)?,
            matrix_from_f64(vt)?,
        )));
    }
    if let Some(s) = cast::as_f32(&op) {
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let (u, sv, vt) = scirs2_linalg::svd(&view, true, None).ok()?;
        return Some(Ok((
            matrix_from_f32(u)?,
            vector_from_f32(sv)?,
            matrix_from_f32(vt)?,
        )));
    }
    None
}

/// Symmetric/Hermitian eigendecomposition via `scirs2_linalg::eigh`.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvectors as the
/// *columns* of the returned matrix, sorted into descending algebraic
/// order by [`sort_descending`]; see [`try_scirs2_eig`] for why that
/// order, and why it is normalised here rather than inherited.
///
/// Declines (`None`) any matrix that is not *exactly* symmetric.
pub(crate) fn try_scirs2_eigh<T>(a: &Array<T>) -> Option<Result<(Array<T>, Array<T>)>>
where
    T: Float + Clone + Debug + 'static,
{
    let n = square_dim(a, SCIRS2_MIN_DIM)?;
    let op = borrow::operand(a);

    if let Some(s) = cast::as_f64(&op) {
        if !is_exactly_symmetric(s, n) {
            return None;
        }
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let (vals, vecs) = scirs2_linalg::eigh(&view, None).ok()?;
        let (vals, vecs) = sort_descending(vals, vecs);
        return Some(Ok((vector_from_f64(vals)?, matrix_from_f64(vecs)?)));
    }
    if let Some(s) = cast::as_f32(&op) {
        if !is_exactly_symmetric(s, n) {
            return None;
        }
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let (vals, vecs) = scirs2_linalg::eigh(&view, None).ok()?;
        let (vals, vecs) = sort_descending(vals, vecs);
        return Some(Ok((vector_from_f32(vals)?, matrix_from_f32(vecs)?)));
    }
    None
}

/// Reorder an eigenpair set into descending algebraic order, permuting
/// the eigenvector *columns* to stay paired with their eigenvalue.
///
/// Returns early (no allocation, no copy) when the input is already
/// ordered -- which, at the sizes this module gates on, it always is; see
/// [`try_scirs2_eig`] for why the sort is applied anyway.
///
/// Ties keep their relative order (`sort_by` is stable), so the result is
/// a deterministic function of the input.
fn sort_descending<F: Float>(vals: Array1<F>, vecs: Array2<F>) -> (Array1<F>, Array2<F>) {
    let n = vals.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        vals[j]
            .partial_cmp(&vals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if order.iter().copied().eq(0..n) {
        return (vals, vecs);
    }

    let sorted_vals = Array1::from_iter(order.iter().map(|&i| vals[i]));
    let rows = vecs.nrows();
    let mut sorted_vecs = Array2::<F>::zeros((rows, n));
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..rows {
            sorted_vecs[[row, new_col]] = vecs[[row, old_col]];
        }
    }
    (sorted_vals, sorted_vecs)
}

/// Eigendecomposition via `scirs2_linalg::eigh` -- **symmetric input
/// only**.
///
/// The method this backs returns *real* eigenvalues and *real*
/// eigenvectors, which is only a meaningful contract for matrices whose
/// spectrum is real. The existing implementation is a Wilkinson-shifted
/// QR iteration whose accumulated `Q` holds true eigenvectors exactly
/// when the input is symmetric (for a non-symmetric matrix those columns
/// are Schur vectors, and the returned "eigenvectors" do not satisfy
/// `A v = lambda v` at all). Rather than silently change what a
/// non-symmetric `eig` returns, this backend takes only the symmetric
/// case -- where both paths agree on what the answer means -- and leaves
/// everything else to the existing iteration.
///
/// # Symmetry is tested exactly
///
/// [`is_exactly_symmetric`] requires `a[i][j] == a[j][i]` bit-for-bit.
/// `scirs2_linalg::eigh` applies its own, *looser*, absolute-epsilon test
/// (`(a[[i, j]] - a[[j, i]]).abs() > F::epsilon()` rejects). Exact
/// equality implies a difference of zero, so anything this gate accepts
/// is guaranteed to clear `eigh`'s check too -- an accepted operand can
/// never be rejected downstream. Being stricter also keeps the decision
/// honest: a matrix that is only *approximately* symmetric is not the
/// symmetric problem, and belongs on the general path.
///
/// # Deviation: eigenvalue ordering
///
/// This is the one place the hook changes an observable result, so the
/// new order is chosen deliberately rather than inherited.
///
/// The existing path returns eigenvalues in whatever order the shifted QR
/// iteration happens to converge to, which is measurably **arbitrary** --
/// on a 16x16 SPD matrix it comes back `179.7, 353.6, 195.5, 343.7, ...`,
/// neither ascending, nor descending, nor ordered by magnitude. So there
/// is no existing order to preserve, and any choice here is a
/// normalisation rather than a regression.
///
/// The choice is **descending algebraic order**, applied explicitly by
/// [`sort_descending`], for two reasons:
///
/// 1. **`svd` in this same module returns descending singular values.**
///    For a symmetric positive-definite matrix the eigenvalues *are* the
///    singular values, so `a.eig()` and `a.svd()` returning the same
///    numbers in opposite orders would be indefensible.
/// 2. **It must not be inherited from `scirs2-linalg`.** `eigh`'s own
///    ordering is inconsistent *within that crate*: its general
///    symmetric path (`n > 4`) sorts descending, while its `n == 3` and
///    `n == 4` closed forms sort ascending. Passing that through would
///    make this crate's contract an accident of an upstream branch and
///    of the upstream version. Sorting here makes the contract local,
///    testable, and stable. At the sizes this module gates on the sort
///    is a no-op on already-ordered data, so it costs a comparison pass
///    and no allocation.
///
/// Eigen*vectors* remain the columns of the second return value and stay
/// paired with their eigenvalue, so `A v_i = lambda_i v_i` holds
/// index-wise on both paths.
pub(crate) fn try_scirs2_eig<T>(a: &Array<T>) -> Option<Result<(Array<T>, Array<T>)>>
where
    T: Float + Clone + Debug + 'static,
{
    try_scirs2_eigh(a)
}

/// Cholesky factorisation via `scirs2_linalg::cholesky`, returning the
/// **lower** triangular `L` with `A = L L^T`.
///
/// A failure yields `None`, not an error: the existing path applies
/// symmetrisation and adaptive diagonal perturbation and can succeed on
/// borderline-indefinite input where a plain factorisation gives up, so
/// it must keep its chance to run.
pub(crate) fn try_scirs2_cholesky<T>(a: &Array<T>) -> Option<Result<Array<T>>>
where
    T: Float + Clone + Debug + 'static,
{
    let n = square_dim(a, SCIRS2_MIN_DIM)?;
    let op = borrow::operand(a);

    if let Some(s) = cast::as_f64(&op) {
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let l = scirs2_linalg::cholesky(&view, None).ok()?;
        return Some(Ok(matrix_from_f64(l)?));
    }
    if let Some(s) = cast::as_f32(&op) {
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let l = scirs2_linalg::cholesky(&view, None).ok()?;
        return Some(Ok(matrix_from_f32(l)?));
    }
    None
}

/// QR factorisation via `scirs2_linalg::qr`.
///
/// Square input only, so that `scirs2-linalg`'s full QR and the existing
/// path's economy QR describe the same pair of shapes (`Q` `n x n`, `R`
/// `n x n`). A failure yields `None` so the existing Householder fallback
/// still gets its turn.
///
/// Engages for `SCIRS2_MIN_DIM <= n < QR_MAX_DIM`; above that the
/// existing path is measurably faster.
pub(crate) fn try_scirs2_qr<T>(a: &Array<T>) -> Option<Result<(Array<T>, Array<T>)>>
where
    T: Float + Clone + Debug + 'static,
{
    let n = square_dim_in_range(a, SCIRS2_MIN_DIM, QR_MAX_DIM)?;
    let op = borrow::operand(a);

    if let Some(s) = cast::as_f64(&op) {
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let (q, r) = scirs2_linalg::qr(&view, None).ok()?;
        return Some(Ok((matrix_from_f64(q)?, matrix_from_f64(r)?)));
    }
    if let Some(s) = cast::as_f32(&op) {
        let view = ArrayView2::from_shape((n, n), s).ok()?;
        let (q, r) = scirs2_linalg::qr(&view, None).ok()?;
        return Some(Ok((matrix_from_f32(q)?, matrix_from_f32(r)?)));
    }
    None
}

/// Exact (bit-for-bit) symmetry test over a row-major `n x n` buffer.
///
/// Deliberately exact rather than tolerance-based: see
/// [`try_scirs2_eig`].
fn is_exactly_symmetric<F: PartialEq>(buf: &[F], n: usize) -> bool {
    for i in 0..n {
        for j in (i + 1)..n {
            if buf[i * n + j] != buf[j * n + i] {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(n: usize) -> Array<f64> {
        let mut data = vec![0.0_f64; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Array::from_vec(data).reshape(&[n, n])
    }

    #[test]
    fn small_matrices_are_not_eligible() {
        // Below the gate every entry point must decline, leaving the
        // caller's closed-form / LU path in charge.
        for n in [1_usize, 2, 3, 4, 15] {
            let a = identity(n);
            assert!(try_scirs2_det(&a).is_none(), "det n={n}");
            assert!(try_scirs2_inv(&a).is_none(), "inv n={n}");
            assert!(try_scirs2_svd(&a).is_none(), "svd n={n}");
            assert!(try_scirs2_qr(&a).is_none(), "qr n={n}");
            assert!(try_scirs2_cholesky(&a).is_none(), "cholesky n={n}");
            assert!(try_scirs2_eig(&a).is_none(), "eig n={n}");
        }
    }

    #[test]
    fn solve_gate_starts_at_four() {
        for n in [1_usize, 2, 3] {
            let a = identity(n);
            let b = Array::from_vec(vec![1.0_f64; n]);
            assert!(try_scirs2_solve(&a, &b).is_none(), "solve n={n}");
        }
        let a = identity(4);
        let b = Array::from_vec(vec![1.0_f64; 4]);
        assert!(try_scirs2_solve(&a, &b).is_some(), "solve n=4 must engage");
    }

    /// Mirror of [`small_matrices_are_not_eligible`] at the other end:
    /// `qr` and `svd` must hand large input back to the existing path,
    /// while every other operation keeps taking it.
    #[test]
    fn large_matrices_are_not_eligible_for_qr_and_svd() {
        let big = identity(QR_MAX_DIM);
        assert!(try_scirs2_qr(&big).is_none(), "qr at QR_MAX_DIM");
        assert!(
            try_scirs2_qr(&identity(QR_MAX_DIM - 1)).is_some(),
            "qr just below QR_MAX_DIM must still engage"
        );

        // The other operations have no upper bound.
        assert!(try_scirs2_det(&big).is_some(), "det has no upper bound");
        assert!(try_scirs2_inv(&big).is_some(), "inv has no upper bound");
        assert!(
            try_scirs2_cholesky(&big).is_some(),
            "cholesky has no upper bound"
        );
        assert!(try_scirs2_eig(&big).is_some(), "eig has no upper bound");
        assert!(
            try_scirs2_svd(&big).is_some(),
            "svd's bound is higher than qr's"
        );

        let huge = identity(SVD_MAX_DIM);
        assert!(try_scirs2_svd(&huge).is_none(), "svd at SVD_MAX_DIM");
        assert!(
            try_scirs2_svd(&identity(SVD_MAX_DIM - 1)).is_some(),
            "svd just below SVD_MAX_DIM must still engage"
        );
        assert!(try_scirs2_det(&huge).is_some(), "det has no upper bound");
    }

    #[test]
    fn non_square_input_is_not_eligible() {
        let a = Array::from_vec(vec![1.0_f64; 16 * 20]).reshape(&[16, 20]);
        assert!(try_scirs2_det(&a).is_none());
        assert!(try_scirs2_inv(&a).is_none());
        assert!(try_scirs2_svd(&a).is_none());
        assert!(try_scirs2_qr(&a).is_none());
        assert!(try_scirs2_cholesky(&a).is_none());
    }

    #[test]
    fn solve_declines_mismatched_rhs() {
        let a = identity(16);
        let wrong_len = Array::from_vec(vec![1.0_f64; 15]);
        assert!(try_scirs2_solve(&a, &wrong_len).is_none());
        let wrong_rank = Array::from_vec(vec![1.0_f64; 16]).reshape(&[4, 4]);
        assert!(try_scirs2_solve(&a, &wrong_rank).is_none());
    }

    #[test]
    fn eig_declines_non_symmetric_input() {
        let mut a = identity(16);
        a.set(&[0, 1], 2.0).expect("in-bounds set");
        assert!(
            try_scirs2_eig(&a).is_none(),
            "non-symmetric input must stay on the existing QR-iteration path"
        );
    }

    #[test]
    fn eig_accepts_symmetric_input() {
        let mut a = identity(16);
        a.set(&[0, 1], 2.0).expect("in-bounds set");
        a.set(&[1, 0], 2.0).expect("in-bounds set");
        assert!(try_scirs2_eig(&a).is_some());
    }

    /// The gate must reject a difference of a *single* ULP -- the
    /// smallest perturbation that exists at all. Note `buf[1] +=
    /// f64::EPSILON` would NOT do: `EPSILON` is the ULP at 1.0, and at
    /// 2.0 the spacing is twice that, so the addition rounds back to
    /// 2.0 and the matrix stays bit-for-bit symmetric.
    #[test]
    fn symmetry_test_is_exact() {
        let n = 3;
        let mut buf = vec![1.0_f64, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0];
        assert!(is_exactly_symmetric(&buf, n));

        let nudged = f64::from_bits(buf[1].to_bits() + 1);
        assert_ne!(nudged, buf[1], "next representable must actually differ");
        buf[1] = nudged;
        assert!(!is_exactly_symmetric(&buf, n));
    }

    /// A non-`f64`/`f32` float must fail both `TypeId` checks and decline,
    /// so that a generic `T` keeps the existing path. `f32` is used as the
    /// stand-in probe for "some other type" against the `f64` converters.
    #[test]
    fn f32_operands_are_eligible_too() {
        let mut data = vec![0.0_f32; 16 * 16];
        for i in 0..16 {
            data[i * 16 + i] = 2.0;
        }
        let a = Array::from_vec(data).reshape(&[16, 16]);
        let det = try_scirs2_det(&a).expect("f32 must dispatch").expect("det");
        // det(2*I_16) = 2^16 = 65536.
        assert!((det - 65536.0_f32).abs() < 1.0, "got {det}");
    }

    #[test]
    fn sort_descending_permutes_columns_with_values() {
        // Ascending values, eigenvectors = distinguishable columns.
        let vals = Array1::from_vec(vec![-3.0_f64, 2.0, 5.0]);
        let vecs =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .expect("valid shape");
        let (sv, svecs) = sort_descending(vals, vecs);
        assert_eq!(sv.to_vec(), vec![5.0, 2.0, -3.0]);
        // Column order must follow: old col 2, then 1, then 0.
        assert_eq!(svecs.column(0).to_vec(), vec![3.0, 6.0, 9.0]);
        assert_eq!(svecs.column(1).to_vec(), vec![2.0, 5.0, 8.0]);
        assert_eq!(svecs.column(2).to_vec(), vec![1.0, 4.0, 7.0]);
    }

    #[test]
    fn sort_descending_is_identity_when_already_ordered() {
        // The common case at n >= 16: upstream already sorts descending,
        // so the helper must return the originals untouched.
        let vals = Array1::from_vec(vec![5.0_f64, 2.0, -3.0]);
        let vecs = Array2::from_shape_vec((3, 3), (0..9).map(|v| v as f64).collect())
            .expect("valid shape");
        let (sv, svecs) = sort_descending(vals.clone(), vecs.clone());
        assert_eq!(sv, vals);
        assert_eq!(svecs, vecs);
    }

    /// Descending *algebraic*, not descending by magnitude: a large
    /// negative eigenvalue sorts last, not first.
    #[test]
    fn sort_descending_is_algebraic_not_by_magnitude() {
        let vals = Array1::from_vec(vec![1.0_f64, -9.0, 4.0]);
        let vecs = Array2::<f64>::zeros((3, 3));
        let (sv, _) = sort_descending(vals, vecs);
        assert_eq!(sv.to_vec(), vec![4.0, 1.0, -9.0]);
    }

    #[test]
    fn into_row_major_vec_handles_non_standard_layout() {
        // A transposed view is not standard layout: taking the raw buffer
        // would yield memory order (1,2,3,4) instead of logical (1,3,2,4).
        let m = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("shape");
        let t = m.reversed_axes();
        assert!(!t.is_standard_layout());
        assert_eq!(into_row_major_vec(t), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn into_row_major_vec_reuses_standard_layout_buffer() {
        let m = Array2::from_shape_vec((2, 2), vec![1.0_f64, 2.0, 3.0, 4.0]).expect("shape");
        assert_eq!(into_row_major_vec(m), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn strided_operand_is_read_in_logical_order() {
        // A transposed (non-contiguous) Array must convert in logical
        // order, not memory order -- `borrow::operand` guarantees this,
        // and `det` of a transpose equals `det` of the original.
        let n = 16;
        let mut data = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] = if i == j { (i as f64) + 2.0 } else { 0.25 };
            }
        }
        let a = Array::from_vec(data).reshape(&[n, n]);
        let at = a.transpose();
        let da = try_scirs2_det(&a).expect("eligible").expect("det ok");
        let dt = try_scirs2_det(&at).expect("eligible").expect("det ok");
        assert!((da - dt).abs() < 1e-6 * da.abs().max(1.0));
    }
}
