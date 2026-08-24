//! [`Series`]: the one generic engine shared by every class in
//! [`super`] -- `Chebyshev<T>`, `Legendre<T>`, `Hermite<T>`, `HermiteE<T>`,
//! `Laguerre<T>`, and `Polynomial<T>` are all `Series<T, SomeBasis>` type
//! aliases (see [`super::define_polynomial_class!`]). Every method here is
//! implemented purely in terms of the small [`super::Basis`] trait, so it
//! never needs to know which concrete family it is operating on.

use super::{add_coefs, mapdomain, mapparms, sub_coefs, trim_coefs, Basis};
use crate::array::Array;
#[cfg(feature = "lapack")]
use crate::error::NumRs2Error;
use crate::error::Result;
use num_traits::Float;
use std::fmt::Debug;
use std::marker::PhantomData;

/// A polynomial expressed as `coef[0]*B_0(x) + coef[1]*B_1(x) + ... +
/// coef[n]*B_n(x)` in some basis `B` (ascending order), together with a
/// `domain`/`window` affine mapping. See the [module docs](super) for the
/// coefficient-order and domain/window conventions.
///
/// Constructed through one of the type aliases in [`super`] (e.g.
/// [`super::Chebyshev`]), never named directly.
#[derive(Clone, Debug)]
pub struct Series<T, B> {
    coef: Vec<T>,
    domain: [T; 2],
    window: [T; 2],
    _marker: PhantomData<B>,
}

impl<T, B> Series<T, B>
where
    T: Float + Debug + 'static,
    B: Basis<T>,
{
    /// Build a series from explicit coefficients, domain, and window. An
    /// empty `coef` is treated as the zero series `[0]` (NumPy rejects an
    /// empty coefficient array outright; defaulting it keeps this
    /// constructor infallible, which matters since every other method
    /// assumes `coef` is never empty).
    pub fn new(coef: Vec<T>, domain: [T; 2], window: [T; 2]) -> Self {
        let coef = if coef.is_empty() {
            vec![T::zero()]
        } else {
            coef
        };
        Series {
            coef,
            domain,
            window,
            _marker: PhantomData,
        }
    }

    /// Build a series using this class's default domain and window (e.g.
    /// `[-1, 1]` for Chebyshev, `[0, 1]` for Laguerre).
    pub fn from_coef(coef: Vec<T>) -> Self {
        Self::new(coef, B::default_domain(), B::default_window())
    }

    /// The series' coefficients, ascending order (`coef()[i]` multiplies the
    /// degree-`i` basis polynomial).
    pub fn coef(&self) -> &[T] {
        &self.coef
    }

    /// The series' domain: the `x`-range `eval`/`call` accept naturally.
    pub fn domain(&self) -> [T; 2] {
        self.domain
    }

    /// The series' window: the range the basis recursion is evaluated over
    /// after `eval`/`call` maps `x` from `domain` into it.
    pub fn window(&self) -> [T; 2] {
        self.window
    }

    /// The series' degree: one less than its coefficient count. Does not
    /// trim trailing zero coefficients first -- call [`Series::trim`] if
    /// that is wanted, matching `numpy.polynomial`'s own `degree()`.
    pub fn degree(&self) -> usize {
        self.coef.len() - 1
    }

    /// Evaluate the series at `x`: maps `x` from `domain` into `window`,
    /// then applies the basis' Clenshaw-style recurrence there. Equivalent
    /// to `numpy.polynomial`'s `__call__`.
    pub fn eval(&self, x: T) -> T {
        let xw = mapdomain(x, self.domain, self.window);
        B::val(xw, &self.coef)
    }

    /// Alias for [`Series::eval`], matching the `call`/`eval` naming used by
    /// the task this module implements; NumPy itself spells this
    /// `__call__`.
    pub fn call(&self, x: T) -> T {
        self.eval(x)
    }

    /// Evaluate the series at every element of `x`.
    pub fn eval_array(&self, x: &Array<T>) -> Array<T> {
        x.map(|v| self.eval(v))
    }

    /// Remove trailing (highest-degree) coefficients with `abs() <= tol`,
    /// keeping at least one coefficient. Mirrors `numpy.polynomial`'s
    /// `trim`.
    pub fn trim(&self, tol: T) -> Self {
        Series::new(trim_coefs(&self.coef, tol), self.domain, self.window)
    }

    /// The `m`-th derivative, still expressed in the same basis over the
    /// same domain/window. When `domain != window`, the chain-rule factor
    /// `scl^m` (from `mapparms(domain, window)`) is applied once at the end
    /// rather than once per derivative step -- valid because each
    /// per-family [`Basis::der_once`] step is linear in its input, so the
    /// two orderings agree exactly, not just approximately.
    pub fn deriv(&self, m: usize) -> Self {
        if m == 0 {
            return self.clone();
        }
        let (_, scl) = mapparms(self.domain, self.window);
        let coef = if m >= self.coef.len() {
            vec![T::zero()]
        } else {
            let mut c = self.coef.clone();
            for _ in 0..m {
                c = B::der_once(&c);
            }
            let scale = scl.powi(m as i32);
            for v in c.iter_mut() {
                *v = *v * scale;
            }
            c
        };
        Series::new(coef, self.domain, self.window)
    }

    /// The `m`-th antiderivative, with integration constants `k` (missing
    /// trailing entries default to zero, matching `numpy.polynomial`'s
    /// `integ`). Each antiderivative's basis-constant term is fixed so the
    /// series evaluates to `k[i]` at domain-space `0`. Unlike [`Series::deriv`],
    /// the per-step scaling by `1/scl` (from `mapparms(domain, window)`)
    /// cannot be hoisted out of the loop: the constant-fixing step
    /// `tmp[0] += k[i] - val(lbnd, tmp)` reads the *already-scaled* `tmp`,
    /// so each of the `m` passes must scale before integrating, not after.
    pub fn integ(&self, m: usize, k: &[T]) -> Self {
        if m == 0 {
            return self.clone();
        }
        let (off, scl) = mapparms(self.domain, self.window);
        // Lower bound is fixed at domain-space 0 (matching NumPy's own
        // `integ` default of `lbnd=None` -> 0); mapped into window space.
        let lbnd_window = off;
        let inv_scl = T::one() / scl;
        let mut k_vals = k.to_vec();
        k_vals.resize(m, T::zero());

        let mut c = self.coef.clone();
        for kv in &k_vals {
            for v in c.iter_mut() {
                *v = *v * inv_scl;
            }
            let mut tmp = B::int_once(&c);
            let correction = *kv - B::val(lbnd_window, &tmp);
            tmp[0] = tmp[0] + correction;
            c = tmp;
        }
        Series::new(c, self.domain, self.window)
    }

    /// Errors if `self` and `other` don't share a domain and window --
    /// mirrors `numpy.polynomial`'s own arithmetic, which raises `TypeError`
    /// on a domain or window mismatch rather than silently reinterpreting
    /// one operand.
    fn check_compatible(&self, other: &Self) -> Result<()> {
        if self.domain != other.domain {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "cannot combine two series with different domains".to_string(),
            ));
        }
        if self.window != other.window {
            return Err(crate::error::NumRs2Error::InvalidOperation(
                "cannot combine two series with different windows".to_string(),
            ));
        }
        Ok(())
    }

    /// `self + other`, in-basis. Errors on a domain/window mismatch (see
    /// [`Series::check_compatible`]).
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.check_compatible(other)?;
        Ok(Series::new(
            add_coefs(&self.coef, &other.coef),
            self.domain,
            self.window,
        ))
    }

    /// `self - other`, in-basis. Errors on a domain/window mismatch (see
    /// [`Series::check_compatible`]).
    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.check_compatible(other)?;
        Ok(Series::new(
            sub_coefs(&self.coef, &other.coef),
            self.domain,
            self.window,
        ))
    }

    /// `self * other`. Errors on a domain/window mismatch (see
    /// [`Series::check_compatible`]).
    ///
    /// Implemented via a **power-basis roundtrip**: both operands convert to
    /// the power basis ([`Basis::to_power`]), multiply there by reusing
    /// [`super::super::core::Polynomial`]'s convolution (genuine reuse of the
    /// existing power-basis type, not a reimplementation), then convert back
    /// ([`Basis::from_power`]). This is simpler and less error-prone than
    /// porting each family's native "generic Clenshaw with series-valued
    /// arithmetic" multiplication (`numpy.polynomial.{legendre,hermite,...}`
    /// each hand-roll a slightly different one), at the cost of an extra
    /// basis-change roundtrip -- an explicitly acceptable tradeoff for this
    /// module. For Chebyshev/Legendre/Hermite/HermiteE/Laguerre specifically
    /// this also loses the numerical-conditioning benefit a native in-basis
    /// product would have kept; prefer a native implementation if that ever
    /// matters for high-degree products.
    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.check_compatible(other)?;
        let p1 = B::to_power(&self.coef);
        let p2 = B::to_power(&other.coef);
        // `core::Polynomial` stores coefficients descending; this module's
        // convention is ascending -- reverse at this one crossing point.
        let d1: Vec<T> = p1.iter().rev().cloned().collect();
        let d2: Vec<T> = p2.iter().rev().cloned().collect();
        let prod = crate::new_modules::polynomial::core::Polynomial::new(d1)
            * crate::new_modules::polynomial::core::Polynomial::new(d2);
        let prod_asc: Vec<T> = prod.coefficients().iter().rev().cloned().collect();
        let coef = B::from_power(&prod_asc);
        Ok(Series::new(coef, self.domain, self.window))
    }

    /// Convert to a different basis `B2`, keeping the same domain/window.
    /// Roundtrips through the power basis ([`Basis::to_power`] then
    /// [`Basis::from_power`]) -- a recurrence-based basis change, exact up to
    /// floating-point rounding (no evaluation/resampling involved).
    pub fn convert<B2: Basis<T>>(&self) -> Series<T, B2> {
        let power = B::to_power(&self.coef);
        let coef2 = B2::from_power(&power);
        Series::<T, B2>::new(coef2, self.domain, self.window)
    }

    /// Least-squares fit of a degree-`deg` series in this basis to `(x, y)`.
    ///
    /// Matches `numpy.polynomial.<Class>.fit(x, y, deg)` with its default
    /// `domain=None, window=None`: `domain` is `[min(x), max(x)]` (widened by
    /// 1 on each side if `x` is constant, so the map stays well-defined),
    /// `window` is this class's default window, and `x` is mapped from
    /// `domain` into `window` before building the pseudo-Vandermonde design
    /// matrix. Columns of that matrix are scaled to unit 2-norm before
    /// solving (via [`crate::new_modules::matrix_decomp::lstsq`], an SVD-based
    /// least-squares solve) and the fitted coefficients rescaled back
    /// afterward -- the same column-scaling `numpy.polynomial.polyutils._fit`
    /// applies, needed because raw Vandermonde-like columns at different
    /// degrees can differ in magnitude by many orders and otherwise dominate
    /// the conditioning of the solve.
    ///
    /// Errors if `x`/`y` aren't 1D, don't share a length, or there are not
    /// more data points than `deg` (mirroring the sibling
    /// [`super::super::fitting::polyfit`]'s behavior, rather than emitting
    /// NumPy's `RankWarning` and returning a rank-deficient fit).
    #[cfg(feature = "lapack")]
    pub fn fit(x: &Array<T>, y: &Array<T>, deg: usize) -> Result<Self> {
        if x.ndim() != 1 || y.ndim() != 1 {
            return Err(NumRs2Error::DimensionMismatch(
                "fit requires 1D x and y arrays".to_string(),
            ));
        }
        let x_vec = x.to_vec();
        let y_vec = y.to_vec();
        let m = x_vec.len();
        if m != y_vec.len() {
            return Err(NumRs2Error::ShapeMismatch {
                expected: x.shape(),
                actual: y.shape(),
            });
        }
        if m == 0 || m <= deg {
            return Err(NumRs2Error::InvalidOperation(format!(
                "fit requires more data points than degree (got {} points for degree {})",
                m, deg
            )));
        }

        let mut lo = x_vec[0];
        let mut hi = x_vec[0];
        for &v in &x_vec[1..] {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let domain = if lo == hi {
            [lo - T::one(), hi + T::one()]
        } else {
            [lo, hi]
        };
        let window = B::default_window();

        let xw: Vec<T> = x_vec
            .iter()
            .map(|&xv| mapdomain(xv, domain, window))
            .collect();

        let ncols = deg + 1;
        let mut vander = vec![T::zero(); m * ncols];
        for (i, &xwi) in xw.iter().enumerate() {
            let row = B::vander_row(xwi, deg);
            vander[i * ncols..i * ncols + ncols].clone_from_slice(&row);
        }

        // Column-scale to unit 2-norm (matches numpy.polynomial.polyutils._fit).
        let mut scl = vec![T::zero(); ncols];
        for (j, scl_j) in scl.iter_mut().enumerate() {
            let mut s = T::zero();
            for i in 0..m {
                let v = vander[i * ncols + j];
                s = s + v * v;
            }
            s = s.sqrt();
            *scl_j = if s == T::zero() { T::one() } else { s };
        }

        let mut scaled = vec![T::zero(); m * ncols];
        for i in 0..m {
            for j in 0..ncols {
                scaled[i * ncols + j] = vander[i * ncols + j] / scl[j];
            }
        }

        let a = Array::from_vec_shape(scaled, &[m, ncols])?;
        let b = Array::from_vec(y_vec);
        let (c_scaled, _residuals, _rank, _singular_values) =
            crate::new_modules::matrix_decomp::lstsq(&a, &b, None)?;
        let c_scaled_vec = c_scaled.to_vec();

        let mut coef = vec![T::zero(); ncols];
        for (j, coef_j) in coef.iter_mut().enumerate() {
            *coef_j = c_scaled_vec[j] / scl[j];
        }

        Ok(Series::new(coef, domain, window))
    }

    /// Roots of the series, as complex values in `domain` space.
    ///
    /// Computed as the eigenvalues of the basis' native companion/colleague
    /// matrix ([`Basis::companion`]) -- built directly in this basis, no
    /// convert-to-power-and-root-find fallback needed for any of the five
    /// classical families or the power basis, since every one of them has a
    /// well-known companion-matrix construction (`numpy.polynomial.
    /// {chebyshev,legendre,hermite,hermite_e,laguerre}` all provide exactly
    /// this) -- computed in window space (where the companion matrix is
    /// built) and mapped back into domain space via
    /// `mapparms(window, domain)` (note: reversed relative to
    /// [`Series::eval`]'s `mapparms(domain, window)`), matching
    /// `numpy.polynomial`'s own `roots()`.
    ///
    /// `self.coef` is trimmed of exactly-zero trailing entries first (NumPy
    /// does not do this, and will divide by zero building the companion
    /// matrix if the leading coefficient is exactly zero); this only ever
    /// removes coefficients that contribute nothing to the evaluated series,
    /// so it cannot change which roots are returned.
    ///
    /// Root accuracy is limited by roundtripping through `f64` inside
    /// [`crate::new_modules::eigenvalues::eigvals`] (a different
    /// implementation path than NumPy's own LAPACK-backed
    /// `numpy.linalg.eigvals`) -- expect agreement with NumPy to roughly
    /// `1e-8`, not full `f64` precision, especially for clustered or
    /// high-multiplicity roots.
    #[cfg(feature = "lapack")]
    pub fn roots(&self) -> Result<Array<scirs2_core::Complex<T>>> {
        let trimmed = trim_coefs(&self.coef, T::zero());
        if trimmed.len() < 2 {
            return Ok(Array::from_vec(vec![]));
        }
        let mat = B::companion(&trimmed)?;
        let n = mat.shape()[0];
        let mut window_roots: Vec<scirs2_core::Complex<T>> = if n == 1 {
            vec![scirs2_core::Complex::new(mat.get(&[0, 0])?, T::zero())]
        } else {
            crate::new_modules::eigenvalues::eigvals(&mat)?.to_vec()
        };
        let (off, scl) = mapparms(self.window, self.domain);
        for z in window_roots.iter_mut() {
            *z = scirs2_core::Complex::new(off + scl * z.re, scl * z.im);
        }
        window_roots.sort_by(|a, b| {
            a.re.partial_cmp(&b.re)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.im.partial_cmp(&b.im).unwrap_or(std::cmp::Ordering::Equal))
        });
        Ok(Array::from_vec(window_roots))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn mapparms_identity_when_domain_equals_window() {
        let (off, scl) = mapparms([-1.0, 1.0], [-1.0, 1.0]);
        assert_relative_eq!(off, 0.0, epsilon = 1e-14);
        assert_relative_eq!(scl, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn mapparms_matches_hand_worked_example() {
        // domain=[0,10] -> window=[-1,1]: off + scl*x, off=-1, scl=0.2
        let (off, scl) = mapparms([0.0, 10.0], [-1.0, 1.0]);
        assert_relative_eq!(off, -1.0, epsilon = 1e-12);
        assert_relative_eq!(scl, 0.2, epsilon = 1e-12);
        assert_relative_eq!(
            mapdomain(0.0, [0.0, 10.0], [-1.0, 1.0]),
            -1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            mapdomain(10.0, [0.0, 10.0], [-1.0, 1.0]),
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            mapdomain(5.0, [0.0, 10.0], [-1.0, 1.0]),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn add_coefs_pads_the_shorter_operand() {
        let out = add_coefs(&[1.0, 2.0, 3.0], &[10.0, 20.0]);
        assert_eq!(out, vec![11.0, 22.0, 3.0]);
    }

    #[test]
    fn sub_coefs_pads_the_shorter_operand() {
        let out = sub_coefs(&[1.0, 2.0, 3.0], &[10.0, 20.0]);
        assert_eq!(out, vec![-9.0, -18.0, 3.0]);
    }

    #[test]
    fn trim_coefs_drops_trailing_near_zero_and_keeps_leading_zero() {
        let out = trim_coefs(&[1.0, 0.0, 1e-15, 0.0], 1e-10);
        assert_eq!(out, vec![1.0, 0.0]);
    }

    #[test]
    fn trim_coefs_of_all_zero_series_returns_single_zero() {
        let out = trim_coefs(&[0.0, 0.0, 0.0], 1e-10);
        assert_eq!(out, vec![0.0]);
    }
}
