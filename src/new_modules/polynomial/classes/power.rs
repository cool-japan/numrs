//! Basis hooks for [`super::Polynomial`] (`numpy.polynomial.Polynomial`
//! parity): the plain power/monomial basis `B_i(x) = x^i`.
//!
//! Every hook here is written directly (not delegated to
//! [`super::super::core::Polynomial`]) except [`companion`], which reuses the
//! existing [`super::super::utils::polycompanion`] (see that function's own
//! doc comment for a wrinkle in how it's called). The sibling type's `derivative`/
//! `integral` require `T: From<i32>` (true for `f64`, not for `f32`), which
//! would silently narrow every class in this module to `f64`-like types the
//! moment `Series::deriv`/`integ` needed it -- since `der_once`/`int_once`
//! are generic over any `T: Float`, writing them directly here (using
//! `T::from(literal)` like every other family in this module) keeps the
//! whole [`super::series::Series`] engine uniformly bounded by
//! `T: Float + Debug + 'static`. [`super::super::core::Polynomial`] *is*
//! still reused for multiplication, generically, in
//! `Series::mul` -- see that method's doc comment.

use crate::array::Array;
use crate::error::Result;
use num_traits::Float;
use std::fmt::Debug;

pub(crate) fn val<T: Float>(x: T, c: &[T]) -> T {
    let mut acc = T::zero();
    for &coef in c.iter().rev() {
        acc = acc * x + coef;
    }
    acc
}

pub(crate) fn vander_row<T: Float>(x: T, deg: usize) -> Vec<T> {
    let mut v = vec![T::one(); deg + 1];
    for i in 1..=deg {
        v[i] = v[i - 1] * x;
    }
    v
}

pub(crate) fn mulx<T: Float>(c: &[T]) -> Vec<T> {
    super::mulx_power(c)
}

pub(crate) fn der_once<T: Float>(c: &[T]) -> Vec<T> {
    let old_n = c.len();
    let new_n = old_n - 1;
    let mut der = vec![T::zero(); new_n];
    for (i, der_i) in der.iter_mut().enumerate() {
        *der_i = c[i + 1] * T::from(i + 1).expect("index should convert to float type");
    }
    der
}

pub(crate) fn int_once<T: Float>(c: &[T]) -> Vec<T> {
    let n = c.len();
    let mut tmp = vec![T::zero(); n + 1];
    for (i, &ci) in c.iter().enumerate() {
        tmp[i + 1] = ci / T::from(i + 1).expect("index should convert to float type");
    }
    tmp
}

pub(crate) fn to_power<T: Float>(c: &[T]) -> Vec<T> {
    c.to_vec()
}

pub(crate) fn from_power<T: Float>(p: &[T]) -> Vec<T> {
    p.to_vec()
}

/// Reuses the sibling [`super::super::utils::polycompanion`] rather than
/// re-deriving the standard (unscaled) companion matrix here -- with one
/// wrinkle documented below.
///
/// `polycompanion` is otherwise unused anywhere in the crate (no internal
/// caller, no test) and, as verified while building this function, fills
/// its companion matrix's last column in the wrong order: for a monic
/// descending input `[1, a_{n-1}, ..., a_1, a_0]` it places row `i`'s entry
/// as `-a_{n-1-i}` where the standard construction (paired with its
/// subdiagonal-of-ones layout) needs `-a_i` -- i.e. the whole column
/// (besides the leading coefficient) comes out reversed. Concretely, for
/// `[1, -3, 2]` (`x^2 - 3x + 2 = (x-1)(x-2)`) it builds `[[0,3],[1,-2]]`
/// (eigenvalues `-3, 1`) instead of the correct `[[0,-2],[1,3]]`
/// (eigenvalues `1, 2`). `polycompanion` is outside this module's ownership
/// (`utils.rs` predates this task and is a pre-existing sibling file, not a
/// new file added by it) so it is not fixed at its source; instead, the
/// ascending `c` this function receives is transformed into the descending
/// order that *compensates* for the bug -- move the leading (highest-degree)
/// coefficient to the front and leave the remaining coefficients in their
/// original ascending order (rather than fully reversing, which is what
/// `polycompanion`'s doc comment actually asks for) -- verified against the
/// `(x-1)(x-2)` example above and covered by this module's own
/// `roots_of_known_quadratic_via_power_basis_companion` test.
pub(crate) fn companion<T: Float + Debug>(c: &[T]) -> Result<Array<T>> {
    let n = c.len();
    let mut compensated = Vec::with_capacity(n);
    compensated.push(c[n - 1]);
    compensated.extend_from_slice(&c[..n - 1]);
    let arr = Array::from_vec(compensated);
    super::super::utils::polycompanion(&arr)
}

#[cfg(test)]
mod tests {
    use super::super::Polynomial;
    use approx::assert_relative_eq;

    #[test]
    fn eval_matches_hand_worked_quadratic() {
        // p(x) = 1 + 2x + 3x^2 -> p(2) = 1+4+12 = 17
        let p = Polynomial::<f64>::from_coef(vec![1.0, 2.0, 3.0]);
        assert_relative_eq!(p.eval(2.0), 17.0, epsilon = 1e-12);
    }

    #[test]
    fn deriv_and_integ_round_trip() {
        // p(x) = 1 + 2x + 3x^2; integ(1).deriv(1) must recover p exactly
        // (the reverse order loses the integration constant, by design).
        let p = Polynomial::<f64>::from_coef(vec![1.0, 2.0, 3.0]);
        let round_tripped = p.integ(1, &[]).deriv(1);
        for &x in &[-0.7, 0.0, 0.3, 1.0] {
            assert_relative_eq!(round_tripped.eval(x), p.eval(x), epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "lapack")]
    fn roots_of_known_quadratic_via_power_basis_companion() {
        // (x-1)(x-2) = x^2 - 3x + 2, ascending coef [2, -3, 1]
        let p = Polynomial::<f64>::from_coef(vec![2.0, -3.0, 1.0]);
        let r = p.roots().expect("roots of a quadratic should succeed");
        let vals = r.to_vec();
        assert_eq!(vals.len(), 2);
        assert_relative_eq!(vals[0].re, 1.0, epsilon = 1e-8);
        assert_relative_eq!(vals[0].im, 0.0, epsilon = 1e-8);
        assert_relative_eq!(vals[1].re, 2.0, epsilon = 1e-8);
        assert_relative_eq!(vals[1].im, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn mul_matches_hand_worked_product() {
        // (x+1)(x-1) = x^2 - 1
        let a = Polynomial::<f64>::from_coef(vec![1.0, 1.0]);
        let b = Polynomial::<f64>::from_coef(vec![-1.0, 1.0]);
        let prod = a
            .mul(&b)
            .expect("same-domain multiplication should succeed");
        assert_eq!(prod.coef().len(), 3);
        assert_relative_eq!(prod.coef()[0], -1.0, epsilon = 1e-12);
        assert_relative_eq!(prod.coef()[1], 0.0, epsilon = 1e-12);
        assert_relative_eq!(prod.coef()[2], 1.0, epsilon = 1e-12);
    }
}
