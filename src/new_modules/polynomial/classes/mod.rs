//! NumPy `numpy.polynomial` class family: [`Chebyshev`], [`Legendre`],
//! [`Hermite`] (physicists'), [`HermiteE`] (probabilists'), [`Laguerre`],
//! and the power-basis [`Polynomial`] (this module's `classes::Polynomial`,
//! distinct from the sibling [`super::core::Polynomial`] -- see its doc
//! comment below).
//!
//! Every class shares one generic engine, [`series::Series`], parameterized
//! by a small [`Basis`] trait that captures exactly the handful of
//! basis-specific formulas each classical orthogonal family needs (three-term
//! recurrence evaluation, one derivative/integral step, the companion-matrix
//! layout, and the change of basis to/from the plain power series). The
//! `define_polynomial_class!` macro wires a family's free functions (in its
//! own small module, e.g. [`chebyshev`]) into a `Basis` impl and produces the
//! public type alias, so `new`/`eval`/`fit`/`roots`/`deriv`/`integ`/`convert`/
//! `trim`/arithmetic are implemented exactly once (in `series.rs`) and shared
//! by every class -- matching NumPy's own `ABCPolyBase` design.
//!
//! # Coefficient order
//!
//! Coefficients here are **ascending**: `coef[0] + coef[1]*B_1(x) + ... +
//! coef[n]*B_n(x)`, matching `numpy.polynomial.Chebyshev` etc. This is the
//! opposite convention from the sibling [`super::core::Polynomial`] (and the
//! rest of this module: `polyfit`/`polyval`/`polyder`/...), which store
//! coefficients **descending** (`numpy.poly1d` style). The two crossing
//! points where this module reaches into the descending-order sibling code
//! (power-basis companion matrix, and the power-basis roundtrip used for
//! multiplication) reverse the slice explicitly at the boundary -- see
//! `power::companion` and `series::Series::mul`.
//!
//! # Domain/window mapping
//!
//! Each class carries a `domain` and `window` (both `[T; 2]`); `eval` maps
//! `x` from `domain` into `window` via the affine map `off + scl*x` before
//! applying the basis recursion (`window` is where the recursion's `x` is
//! assumed to range, e.g. `[-1, 1]` for Chebyshev/Legendre/Hermite/HermiteE).
//! Default domain equals default window for every class except [`Laguerre`]
//! (domain `[0, 1]`, matching `numpy.polynomial.Laguerre`); this makes the
//! default mapping the identity for everything except Laguerre, so most
//! tests exercise `scl = 1` -- at least one `fit` test uses a non-default
//! data range specifically to exercise `scl != 1`.
//!
//! # What is reused vs. reimplemented
//!
//! The task that produced this module pointed at `orthogonal.rs` for
//! "existing evaluation routines" to reuse; the actual layout (reported once,
//! here) is:
//! - `src/new_modules/special/orthogonal.rs` has Legendre/associated-Legendre,
//!   spherical harmonics, Airy, and Struve -- no Chebyshev/Hermite/Laguerre.
//! - `src/new_modules/polynomial/special.rs`'s `OrthogonalPolynomials` has the
//!   Chebyshev/Legendre/Hermite/Laguerre/Jacobi three-term recurrences, but
//!   as *symbolic* generators returning a power-basis [`super::core::Polynomial`]
//!   for one fixed degree `n` -- not something a `fit`/`call` hot path can
//!   reuse directly without rebuilding an O(n) polynomial-arithmetic chain
//!   per evaluated point.
//!
//! This module therefore applies the *same* three-term recurrences
//! numerically (`val`/`vander_row`/`der_once`/`int_once` below, ported
//! directly from `numpy.polynomial.{chebyshev,legendre,hermite,hermite_e,
//! laguerre}`, which use this exact approach rather than routing through a
//! symbolic power-basis form) instead of reimplementing a divergent formula.
//! `mul` and `convert` *do* roundtrip through the power basis (documented at
//! their definitions), which is the one place `super::core::Polynomial` is
//! reused directly, and is the explicitly-sanctioned simplification for
//! multiplication in a non-native basis.

pub mod chebyshev;
pub mod hermite;
pub mod hermite_e;
pub mod laguerre;
pub mod legendre;
pub mod power;
pub mod series;

use crate::array::Array;
use crate::error::Result;
use num_traits::Float;
use std::fmt::Debug;

/// The handful of basis-specific formulas [`series::Series`] needs from a
/// classical orthogonal polynomial family (or the plain power basis) to
/// implement the full shared interface generically.
///
/// All coefficient slices are **ascending order** (`c[0]` is the coefficient
/// of the degree-0 basis polynomial). Implementations are provided by
/// `define_polynomial_class!`, which wires each method to a same-named free
/// function in that family's own module.
///
/// Requires `Self: Clone + Copy + Debug + Default` (trivial for the
/// zero-sized markers every impl here uses) so that
/// [`series::Series<T, B>`](series::Series) can itself derive `Clone`/`Debug`
/// generically over `B` without every method that needs to clone a `Series`
/// (e.g. [`series::Series::deriv`] and [`series::Series::integ`] on `m == 0`)
/// having to repeat that bound.
pub trait Basis<T>: Clone + Copy + Debug + Default {
    /// Default `domain` for this class (e.g. `[-1, 1]` for Chebyshev).
    fn default_domain() -> [T; 2];
    /// Default `window` for this class (equal to `default_domain()` for
    /// every class except [`Laguerre`]).
    fn default_window() -> [T; 2];

    /// Evaluate `sum_i c[i] * B_i(x)` via the family's Clenshaw-style
    /// backward recurrence. `x` is already mapped into window space.
    fn val(x: T, c: &[T]) -> T;

    /// Row `[B_0(x), B_1(x), ..., B_deg(x)]` of the pseudo-Vandermonde
    /// matrix, built via the family's forward recurrence. `x` is already
    /// mapped into window space.
    fn vander_row(x: T, deg: usize) -> Vec<T>;

    /// Multiply an ascending-order basis series by the independent variable
    /// `x`, staying in-basis (length grows by one).
    fn mulx(c: &[T]) -> Vec<T>;

    /// One derivative pass (ascending order), producing a series of length
    /// `c.len() - 1`. Callers guarantee `c.len() >= 2`; the `scl` chain-rule
    /// factor from `mapparms` is applied by the caller, not here.
    fn der_once(c: &[T]) -> Vec<T>;

    /// One antiderivative pass *ignoring* the integration constant (index 0
    /// of the result is left however the family's recurrence naturally
    /// produces it -- some are exactly zero, Laguerre's is not), producing a
    /// series of length `c.len() + 1`. The caller fixes up index 0 using
    /// [`Basis::val`] so the antiderivative evaluates to the requested
    /// constant at the mapped lower bound.
    fn int_once(c: &[T]) -> Vec<T>;

    /// Build the (unrotated) companion/colleague matrix for the trimmed,
    /// ascending series `c` (`c.len() >= 2`, `c.last() != 0`). Eigenvalues of
    /// this matrix are the roots of the series, in window space.
    fn companion(c: &[T]) -> Result<Array<T>>;

    /// Convert this basis' ascending coefficients to the plain power basis
    /// (ascending): `sum_i c[i] * B_i(x) == sum_i to_power(c)[i] * x^i`.
    fn to_power(c: &[T]) -> Vec<T>;

    /// Convert plain power-basis ascending coefficients into this basis
    /// (inverse of [`Basis::to_power`]).
    fn from_power(p: &[T]) -> Vec<T>;
}

/// Linear map parameters `(off, scl)` such that `off + scl*x` sends `old[0]
/// -> new[0]` and `old[1] -> new[1]`. Mirrors `numpy.polynomial.polyutils.
/// mapparms`.
pub(crate) fn mapparms<T: Float>(old: [T; 2], new: [T; 2]) -> (T, T) {
    let old_len = old[1] - old[0];
    let new_len = new[1] - new[0];
    let off = (old[1] * new[0] - old[0] * new[1]) / old_len;
    let scl = new_len / old_len;
    (off, scl)
}

/// Apply the affine map from `old` to `new` to a single point. Mirrors
/// `numpy.polynomial.polyutils.mapdomain`.
pub(crate) fn mapdomain<T: Float>(x: T, old: [T; 2], new: [T; 2]) -> T {
    let (off, scl) = mapparms(old, new);
    off + scl * x
}

/// Elementwise sum of two ascending-order coefficient slices, zero-padded to
/// the longer length. Basis-agnostic: every classical family (and the power
/// basis) represents a linear combination of basis functions the same way,
/// so addition never needs a per-family hook.
pub(crate) fn add_coefs<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    let n = a.len().max(b.len());
    let mut out = vec![T::zero(); n];
    for (i, &v) in a.iter().enumerate() {
        out[i] = out[i] + v;
    }
    for (i, &v) in b.iter().enumerate() {
        out[i] = out[i] + v;
    }
    out
}

/// Elementwise difference `a - b` of two ascending-order coefficient slices,
/// zero-padded to the longer length. See [`add_coefs`].
pub(crate) fn sub_coefs<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    let n = a.len().max(b.len());
    let mut out = vec![T::zero(); n];
    for (i, &v) in a.iter().enumerate() {
        out[i] = out[i] + v;
    }
    for (i, &v) in b.iter().enumerate() {
        out[i] = out[i] - v;
    }
    out
}

/// Multiply an ascending-order **power-basis** series by `x` (prepend a zero
/// coefficient). Used by every orthogonal family's `to_power` conversion,
/// which accumulates its result directly in the power basis (mirroring
/// `numpy.polynomial.polynomial.polymulx` as used by `cheb2poly`/`leg2poly`/
/// etc.) -- distinct from each family's own in-basis [`Basis::mulx`].
pub(crate) fn mulx_power<T: Float>(c: &[T]) -> Vec<T> {
    let mut out = vec![T::zero(); c.len() + 1];
    out[1..].clone_from_slice(c);
    out
}

/// Remove trailing (highest-degree) coefficients with `abs() <= tol`.
/// Mirrors `numpy.polynomial.polyutils.trimcoef`: if every coefficient is
/// trimmed away, returns `[0]` rather than an empty series.
pub(crate) fn trim_coefs<T: Float>(c: &[T], tol: T) -> Vec<T> {
    let mut last_significant = None;
    for (i, v) in c.iter().enumerate() {
        if v.abs() > tol {
            last_significant = Some(i);
        }
    }
    match last_significant {
        Some(i) => c[..=i].to_vec(),
        None => vec![T::zero()],
    }
}

/// Defines one polynomial class: a zero-sized [`Basis`] marker wired to the
/// free functions in `$module` (an absolute path, e.g.
/// `crate::new_modules::polynomial::classes::chebyshev`), plus the public
/// `$name<T> = Series<T, $marker>` type alias. `$module` must export `val`,
/// `vander_row`, `mulx`, `der_once`, `int_once`, `companion`, `to_power`, and
/// `from_power` with the signatures declared on [`Basis`].
///
/// A macro (rather than one impl per family written by hand) is what lets
/// all six classes share the *exact* same `new`/`eval`/`fit`/`roots`/`deriv`/
/// `integ`/`convert`/`trim`/arithmetic surface implemented in `series.rs`:
/// each family only ever supplies its handful of numeric formulas.
macro_rules! define_polynomial_class {
    (
        $(#[$meta:meta])*
        $name:ident, $marker:ident, $module:ident,
        domain = [$dlo:expr, $dhi:expr],
        window = [$wlo:expr, $whi:expr]
    ) => {
        /// Zero-sized [`Basis`](crate::new_modules::polynomial::classes::Basis)
        /// marker wiring this family's free functions into the shared
        /// [`Series`](crate::new_modules::polynomial::classes::series::Series) engine.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct $marker;

        impl<T> crate::new_modules::polynomial::classes::Basis<T> for $marker
        where
            T: num_traits::Float + std::fmt::Debug + 'static,
        {
            fn default_domain() -> [T; 2] {
                [
                    T::from($dlo).expect("domain bound literal should convert to float type"),
                    T::from($dhi).expect("domain bound literal should convert to float type"),
                ]
            }

            fn default_window() -> [T; 2] {
                [
                    T::from($wlo).expect("window bound literal should convert to float type"),
                    T::from($whi).expect("window bound literal should convert to float type"),
                ]
            }

            fn val(x: T, c: &[T]) -> T {
                crate::new_modules::polynomial::classes::$module::val(x, c)
            }

            fn vander_row(x: T, deg: usize) -> Vec<T> {
                crate::new_modules::polynomial::classes::$module::vander_row(x, deg)
            }

            fn mulx(c: &[T]) -> Vec<T> {
                crate::new_modules::polynomial::classes::$module::mulx(c)
            }

            fn der_once(c: &[T]) -> Vec<T> {
                crate::new_modules::polynomial::classes::$module::der_once(c)
            }

            fn int_once(c: &[T]) -> Vec<T> {
                crate::new_modules::polynomial::classes::$module::int_once(c)
            }

            fn companion(
                c: &[T],
            ) -> crate::error::Result<crate::array::Array<T>> {
                crate::new_modules::polynomial::classes::$module::companion(c)
            }

            fn to_power(c: &[T]) -> Vec<T> {
                crate::new_modules::polynomial::classes::$module::to_power(c)
            }

            fn from_power(p: &[T]) -> Vec<T> {
                crate::new_modules::polynomial::classes::$module::from_power(p)
            }
        }

        $(#[$meta])*
        pub type $name<T> =
            crate::new_modules::polynomial::classes::series::Series<T, $marker>;
    };
}

define_polynomial_class!(
    /// NumPy-parity Chebyshev series: `sum_i coef[i] * T_i(x)`, domain and
    /// window both defaulting to `[-1, 1]`.
    Chebyshev, ChebyshevBasis, chebyshev,
    domain = [-1.0, 1.0], window = [-1.0, 1.0]
);

define_polynomial_class!(
    /// NumPy-parity Legendre series: `sum_i coef[i] * P_i(x)`, domain and
    /// window both defaulting to `[-1, 1]`.
    Legendre, LegendreBasis, legendre,
    domain = [-1.0, 1.0], window = [-1.0, 1.0]
);

define_polynomial_class!(
    /// NumPy-parity physicists' Hermite series: `sum_i coef[i] * H_i(x)`,
    /// domain and window both defaulting to `[-1, 1]`.
    Hermite, HermiteBasis, hermite,
    domain = [-1.0, 1.0], window = [-1.0, 1.0]
);

define_polynomial_class!(
    /// NumPy-parity probabilists' Hermite series: `sum_i coef[i] * He_i(x)`,
    /// domain and window both defaulting to `[-1, 1]`.
    HermiteE, HermiteEBasis, hermite_e,
    domain = [-1.0, 1.0], window = [-1.0, 1.0]
);

define_polynomial_class!(
    /// NumPy-parity Laguerre series: `sum_i coef[i] * L_i(x)`, domain and
    /// window both defaulting to `[0, 1]` (matching
    /// `numpy.polynomial.Laguerre`, unlike every other class here).
    Laguerre, LaguerreBasis, laguerre,
    domain = [0.0, 1.0], window = [0.0, 1.0]
);

define_polynomial_class!(
    /// NumPy-parity power-basis series `sum_i coef[i] * x^i` with
    /// domain/window support -- the `numpy.polynomial.Polynomial` analogue.
    /// Distinct from [`super::core::Polynomial`] (this module's pre-existing
    /// `numpy.poly1d`-style type: descending coefficients, no domain/window).
    /// Not re-exported as a bare `Polynomial` from the parent `polynomial`
    /// module to avoid colliding with that sibling type -- reach this one as
    /// `classes::Polynomial`.
    Polynomial, PowerBasis, power,
    domain = [-1.0, 1.0], window = [-1.0, 1.0]
);
