//! Basis hooks for [`super::Polynomial`] (`numpy.polynomial.Polynomial`
//! parity): the plain power/monomial basis `B_i(x) = x^i`.
//!
//! Every hook here is written directly (not delegated to
//! [`super::super::core::Polynomial`]) except `companion`, which reuses the
//! existing [`super::super::utils::polycompanion`] (a plain ascending-to-
//! descending reversal bridges the two conventions; see that function's own
//! doc comment). The sibling type's `derivative`/
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
/// re-deriving the standard (unscaled) companion matrix here.
///
/// `polycompanion` takes coefficients in descending order (`c[0]` the
/// leading/highest-degree term), while this trait method receives `c` in
/// this module's ascending order (`c[0]` the constant term, `c.last()` the
/// leading term, per [`super::Basis::companion`]'s contract) -- so the
/// conversion is a straight reversal, no reordering trick needed.
pub(crate) fn companion<T: Float + Debug>(c: &[T]) -> Result<Array<T>> {
    let descending: Vec<T> = c.iter().rev().copied().collect();
    let arr = Array::from_vec(descending);
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
