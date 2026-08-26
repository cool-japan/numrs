//! Local thin Householder QR that keeps its reflectors *implicit*.
//!
//! # Why not `scirs2_linalg::qr`
//!
//! Every node of a TSQR tree needs two things from its local
//! factorization: the small `R`, and the ability to apply `Q` (or `Q^T`) to
//! a *different* matrix later, without ever materializing `Q`.
//! [`scirs2_linalg::qr`] gives neither cheaply — it returns an explicit
//! `m x m` `Q`, which is the whole cost TSQR exists to avoid (an `m x n`
//! leaf block would carry an `m x m` factor), and it rejects `m < n`
//! outright. So this module implements the LAPACK `geqr2`/`larf` pair
//! directly:
//!
//! - `A` is overwritten in place: the upper triangle becomes `R`, and the
//!   strictly lower triangle holds the reflector vectors `v_j` with the
//!   implicit unit diagonal (`v_j[j] = 1`) never stored.
//! - `tau[j]` completes reflector `j`: `H_j = I - tau_j v_j v_j^T`.
//! - `Q = H_0 H_1 ... H_{n-1}`, applied by
//!   [`HouseholderQr::apply_q_in_place`] / [`HouseholderQr::apply_qt_in_place`]
//!   in `O(m n k)` without ever forming `Q`.
//!
//! One factorization is therefore `O(m n)` storage — the same array that
//! came in — which is what makes the "store *all* tree factors" strategy in
//! [`mod@super::tsqr`] affordable.
//!
//! # Thin-Q identity used everywhere upstream
//!
//! For `m >= n`, applying the full `m x m` `Q` to `[D; 0]` (a `k`-column
//! `D` on top, zeros below) yields exactly `Q_thin * D`, and the first `n`
//! rows of `Q^T B` are exactly `Q_thin^T B`. Both directions are used by
//! [`mod@super::tsqr`], which never forms a thin `Q` explicitly.

use super::matrix::DistFloat;
use super::DistributedLinalgError;
use scirs2_core::ndarray::{Array2, ArrayView2};

/// A thin QR factorization holding its reflectors implicitly.
///
/// Produced by [`HouseholderQr::factor`]; `m >= n` is required (a wide
/// block needs a different algorithm entirely, not a fallback — see
/// [`mod@super::tsqr`]).
#[derive(Debug, Clone)]
pub struct HouseholderQr<T> {
    /// `m x n`: `R` in the upper triangle, reflector tails below it.
    factors: Array2<T>,
    /// One `tau` per column.
    tau: Vec<T>,
}

impl<T: DistFloat> HouseholderQr<T> {
    /// Factor `a` (`m x n`, `m >= n`) into implicit reflectors plus `R`.
    ///
    /// # Errors
    ///
    /// [`DistributedLinalgError::UnsupportedShape`] when `m < n`: this
    /// routine deliberately refuses the wide case rather than silently
    /// producing a rank-deficient answer.
    pub fn factor(a: Array2<T>) -> Result<Self, DistributedLinalgError> {
        let (m, n) = a.dim();
        if m < n {
            return Err(DistributedLinalgError::UnsupportedShape(format!(
                "thin Householder QR needs at least as many rows as columns, got {m}x{n}"
            )));
        }

        let mut factors = a;
        let mut tau = vec![T::zero(); n];
        for j in 0..n {
            let alpha = factors[[j, j]];

            // ||x||_2 of the sub-diagonal tail, computed without squaring
            // into an overflow: scale by the largest magnitude first.
            let mut max_abs = T::zero();
            for i in (j + 1)..m {
                let abs = factors[[i, j]].abs();
                if abs > max_abs {
                    max_abs = abs;
                }
            }
            if max_abs == T::zero() {
                // Column is already zero below the diagonal: H_j = I.
                tau[j] = T::zero();
                continue;
            }
            let mut scaled_sum = T::zero();
            for i in (j + 1)..m {
                let scaled = factors[[i, j]] / max_abs;
                scaled_sum += scaled * scaled;
            }
            let xnorm = max_abs * scaled_sum.sqrt();

            // beta = -sign(alpha) * ||[alpha; x]||, chosen so that
            // alpha - beta never cancels.
            let norm = alpha.hypot(xnorm);
            let beta = if alpha < T::zero() { norm } else { -norm };
            let scale = alpha - beta;
            if scale == T::zero() {
                tau[j] = T::zero();
                continue;
            }
            tau[j] = (beta - alpha) / beta;
            for i in (j + 1)..m {
                factors[[i, j]] /= scale;
            }
            factors[[j, j]] = beta;

            // Apply H_j to the trailing columns, in place.
            let tau_j = tau[j];
            for c in (j + 1)..n {
                let mut dot = factors[[j, c]];
                for i in (j + 1)..m {
                    dot += factors[[i, j]] * factors[[i, c]];
                }
                let w = tau_j * dot;
                factors[[j, c]] -= w;
                for i in (j + 1)..m {
                    let v = factors[[i, j]];
                    factors[[i, c]] -= w * v;
                }
            }
        }

        Ok(Self { factors, tau })
    }

    /// Rows of the factored block.
    pub fn nrows(&self) -> usize {
        self.factors.nrows()
    }

    /// Columns of the factored block.
    pub fn ncols(&self) -> usize {
        self.factors.ncols()
    }

    /// The `n x n` upper triangular factor `R`.
    pub fn r(&self) -> Array2<T> {
        let n = self.ncols();
        let mut r = Array2::<T>::zeros((n, n));
        for i in 0..n {
            for j in i..n {
                r[[i, j]] = self.factors[[i, j]];
            }
        }
        r
    }

    /// Raw access to the packed factors (`R` above the diagonal, reflector
    /// tails below).
    pub fn packed(&self) -> ArrayView2<'_, T> {
        self.factors.view()
    }

    /// The `tau` scalars, one per reflector.
    pub fn tau(&self) -> &[T] {
        &self.tau
    }

    /// Apply reflector `j` to every column of `b` (`m x k`), in place.
    ///
    /// `H_j` is symmetric, so this is used by both directions; only the
    /// order the reflectors are visited in differs.
    fn apply_reflector(&self, j: usize, b: &mut Array2<T>) {
        let tau_j = self.tau[j];
        if tau_j == T::zero() {
            return;
        }
        let m = self.nrows();
        let k = b.ncols();
        for c in 0..k {
            let mut dot = b[[j, c]];
            for i in (j + 1)..m {
                dot += self.factors[[i, j]] * b[[i, c]];
            }
            let w = tau_j * dot;
            b[[j, c]] -= w;
            for i in (j + 1)..m {
                let v = self.factors[[i, j]];
                b[[i, c]] -= w * v;
            }
        }
    }

    fn check_rows(&self, b: &Array2<T>) -> Result<(), DistributedLinalgError> {
        if b.nrows() != self.nrows() {
            return Err(DistributedLinalgError::DimensionMismatch(format!(
                "operand has {} rows, factorization has {}",
                b.nrows(),
                self.nrows()
            )));
        }
        Ok(())
    }

    /// `b := Q^T b`, with `Q = H_0 ... H_{n-1}` so `Q^T = H_{n-1} ... H_0`.
    ///
    /// The first `n` rows of the result are `Q_thin^T b`.
    pub fn apply_qt_in_place(&self, b: &mut Array2<T>) -> Result<(), DistributedLinalgError> {
        self.check_rows(b)?;
        for j in 0..self.ncols() {
            self.apply_reflector(j, b);
        }
        Ok(())
    }

    /// `b := Q b`.
    ///
    /// With zeros below row `n`, the result is `Q_thin * b[..n]`.
    pub fn apply_q_in_place(&self, b: &mut Array2<T>) -> Result<(), DistributedLinalgError> {
        self.check_rows(b)?;
        for j in (0..self.ncols()).rev() {
            self.apply_reflector(j, b);
        }
        Ok(())
    }

    /// The explicit `m x n` thin `Q`, formed by applying `Q` to `[I_n; 0]`.
    ///
    /// Only for tests, diagnostics, and the final materialization step of
    /// [`super::decomp::distributed_qr`] — the algorithms themselves apply
    /// `Q` implicitly.
    pub fn thin_q(&self) -> Result<Array2<T>, DistributedLinalgError> {
        let (m, n) = (self.nrows(), self.ncols());
        let mut q = Array2::<T>::zeros((m, n));
        for i in 0..n {
            q[[i, i]] = T::one();
        }
        self.apply_q_in_place(&mut q)?;
        Ok(q)
    }
}

#[cfg(test)]
mod tests {
    use super::super::matrix::testutil::{deterministic_matrix, frobenius};
    use super::*;
    use scirs2_core::ndarray::{s, Array2};

    /// `R` is unique only up to the sign of each row: fix the convention by
    /// making every diagonal entry positive before comparing two `R`s.
    fn normalize_row_signs(r: &mut Array2<f64>) {
        let n = r.nrows().min(r.ncols());
        for i in 0..n {
            if r[[i, i]] < 0.0 {
                for j in 0..r.ncols() {
                    r[[i, j]] = -r[[i, j]];
                }
            }
        }
    }

    fn identity(n: usize) -> Array2<f64> {
        let mut eye = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            eye[[i, i]] = 1.0;
        }
        eye
    }

    #[test]
    fn square_r_matches_scirs2_qr_up_to_row_signs() {
        for n in [1usize, 2, 3, 5, 8] {
            let a = deterministic_matrix(n, n, 7 + n as u64);
            let qr = HouseholderQr::factor(a.clone()).expect("factors");
            let mut mine = qr.r();
            let (_, reference_r) = scirs2_linalg::qr(&a.view(), None).expect("scirs2 qr");
            let mut reference = reference_r.slice(s![..n, ..n]).to_owned();

            normalize_row_signs(&mut mine);
            normalize_row_signs(&mut reference);
            let diff = &mine - &reference;
            assert!(
                frobenius(&diff.view()) < 1e-10,
                "n={n}: ||R_mine - R_scirs2||_F = {}",
                frobenius(&diff.view())
            );
        }
    }

    #[test]
    fn tall_factorization_reconstructs_the_input() {
        for (m, n) in [(8usize, 3usize), (32, 4), (17, 17), (5, 1)] {
            let a = deterministic_matrix(m, n, 100 + m as u64);
            let qr = HouseholderQr::factor(a.clone()).expect("factors");
            let q = qr.thin_q().expect("thin q");
            let r = qr.r();

            let reconstructed = q.dot(&r);
            let diff = &reconstructed - &a;
            assert!(
                frobenius(&diff.view()) < 1e-10,
                "{m}x{n}: ||QR - A||_F = {}",
                frobenius(&diff.view())
            );

            let qtq = q.t().dot(&q);
            let ortho = &qtq - &identity(n);
            assert!(
                frobenius(&ortho.view()) < 1e-10,
                "{m}x{n}: ||Q^T Q - I||_F = {}",
                frobenius(&ortho.view())
            );

            // R is upper triangular.
            for i in 0..n {
                for j in 0..i {
                    assert!(r[[i, j]].abs() < 1e-14, "R[{i},{j}] = {}", r[[i, j]]);
                }
            }
        }
    }

    #[test]
    fn apply_qt_then_apply_q_is_the_identity() {
        let (m, n, k) = (12usize, 4usize, 3usize);
        let a = deterministic_matrix(m, n, 55);
        let qr = HouseholderQr::factor(a).expect("factors");
        let b = deterministic_matrix(m, k, 66);

        let mut round_trip = b.clone();
        qr.apply_qt_in_place(&mut round_trip).expect("Q^T b");
        qr.apply_q_in_place(&mut round_trip).expect("Q Q^T b");
        let diff = &round_trip - &b;
        assert!(frobenius(&diff.view()) < 1e-12);
    }

    #[test]
    fn implicit_and_explicit_thin_q_agree() {
        let (m, n, k) = (10usize, 3usize, 2usize);
        let a = deterministic_matrix(m, n, 77);
        let qr = HouseholderQr::factor(a).expect("factors");
        let q = qr.thin_q().expect("thin q");
        let d = deterministic_matrix(n, k, 88);

        // Q_thin * D, computed implicitly by padding with zero rows.
        let mut padded = Array2::<f64>::zeros((m, k));
        padded.slice_mut(s![..n, ..]).assign(&d);
        qr.apply_q_in_place(&mut padded).expect("Q [D; 0]");

        let explicit = q.dot(&d);
        let diff = &padded - &explicit;
        assert!(
            frobenius(&diff.view()) < 1e-12,
            "||Q[D;0] - Q_thin D||_F = {}",
            frobenius(&diff.view())
        );

        // And the transpose direction: top n rows of Q^T B equal Q_thin^T B.
        let b = deterministic_matrix(m, k, 99);
        let mut qt_b = b.clone();
        qr.apply_qt_in_place(&mut qt_b).expect("Q^T B");
        let explicit_t = q.t().dot(&b);
        let diff_t = &qt_b.slice(s![..n, ..]).to_owned() - &explicit_t;
        assert!(frobenius(&diff_t.view()) < 1e-12);
    }

    #[test]
    fn zero_column_leaves_a_zero_reflector() {
        let mut a = Array2::<f64>::zeros((4, 2));
        a[[0, 0]] = 0.0;
        a[[0, 1]] = 1.0;
        let qr = HouseholderQr::factor(a.clone()).expect("factors");
        // First column is entirely zero: tau_0 must be zero, R[0,0] zero.
        assert_eq!(qr.tau()[0], 0.0);
        assert_eq!(qr.r()[[0, 0]], 0.0);
        let reconstructed = qr.thin_q().expect("thin q").dot(&qr.r());
        let diff = &reconstructed - &a;
        assert!(frobenius(&diff.view()) < 1e-12);
    }

    #[test]
    fn wide_block_is_refused_not_approximated() {
        let a = deterministic_matrix(2, 5, 5);
        assert!(matches!(
            HouseholderQr::factor(a),
            Err(DistributedLinalgError::UnsupportedShape(_))
        ));
    }
}
