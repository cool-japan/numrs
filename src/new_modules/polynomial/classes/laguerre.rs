//! Basis hooks for [`super::Laguerre`] (`numpy.polynomial.Laguerre`
//! parity): Laguerre polynomials `L_i(x)`.
//!
//! Ported directly from `numpy.polynomial.laguerre` (`lagval`, `lagvander`,
//! `lagmulx`, `lagder`, `lagint`, `lagcompanion`, `lag2poly`, `poly2lag`),
//! applying the recurrence `L_0=1, L_1=1-x,
//! (n+1)L_{n+1}=(2n+1-x)L_n - n*L_{n-1}` numerically -- see
//! [`super::chebyshev`]'s module doc comment for why this (rather than
//! routing through `special::OrthogonalPolynomials::laguerre`'s symbolic
//! power-basis generator) is the reuse this task called for.
//!
//! Unlike every other family in this module, Laguerre's default domain is
//! `[0, 1]`, not `[-1, 1]` (matching `numpy.polynomial.Laguerre`) -- the
//! window is still `[0, 1]` too, so the default domain->window map is still
//! the identity, but the mapped `x` the recurrences see is expected to be
//! non-negative-ish rather than confined to `[-1, 1]`.

use super::{add_coefs, mulx_power, sub_coefs};
use crate::array::Array;
use crate::error::Result;
use num_traits::Float;
use std::fmt::Debug;

pub(crate) fn val<T: Float>(x: T, c: &[T]) -> T {
    let n = c.len();
    if n == 0 {
        return T::zero();
    }
    if n == 1 {
        return c[0];
    }
    if n == 2 {
        return c[0] + c[1] * x;
    }
    let mut nd = n;
    let mut c0 = c[n - 2];
    let mut c1 = c[n - 1];
    for i in 3..=n {
        let tmp = c0;
        nd -= 1;
        let ndt = T::from(nd).expect("index should convert to float type");
        c0 = c[n - i] - (c1 * T::from(nd - 1).expect("index should convert to float type")) / ndt;
        c1 = tmp
            + (c1 * (T::from(2 * nd - 1).expect("index should convert to float type") - x)) / ndt;
    }
    c0 + c1 * (T::one() - x)
}

pub(crate) fn vander_row<T: Float>(x: T, deg: usize) -> Vec<T> {
    let mut v = vec![T::zero(); deg + 1];
    v[0] = T::one();
    if deg > 0 {
        v[1] = T::one() - x;
        for i in 2..=deg {
            let it = T::from(i).expect("index should convert to float type");
            let factor = T::from(2 * i - 1).expect("index should convert to float type") - x;
            v[i] = (v[i - 1] * factor
                - v[i - 2] * T::from(i - 1).expect("index should convert to float type"))
                / it;
        }
    }
    v
}

pub(crate) fn mulx<T: Float>(c: &[T]) -> Vec<T> {
    let n = c.len();
    if n == 1 && c[0] == T::zero() {
        return c.to_vec();
    }
    let mut prd = vec![T::zero(); n + 1];
    prd[0] = c[0];
    prd[1] = -c[0];
    for i in 1..n {
        let it = T::from(i).expect("index should convert to float type");
        prd[i + 1] = -c[i] * T::from(i + 1).expect("index should convert to float type");
        prd[i] = prd[i] + c[i] * T::from(2 * i + 1).expect("index should convert to float type");
        prd[i - 1] = prd[i - 1] - c[i] * it;
    }
    prd
}

/// One derivative step. Callers guarantee `c.len() >= 2`.
pub(crate) fn der_once<T: Float>(c: &[T]) -> Vec<T> {
    let old_n = c.len();
    let new_n = old_n - 1;
    let mut cw = c.to_vec();
    let mut der = vec![T::zero(); new_n];
    let mut j = new_n;
    while j > 1 {
        der[j - 1] = -cw[j];
        cw[j - 1] = cw[j - 1] + cw[j];
        j -= 1;
    }
    der[0] = -cw[1];
    der
}

/// One antiderivative step, ignoring the integration constant. Unlike every
/// other family here, Laguerre's raw recurrence leaves index 0 at `c[0]`
/// (not zero); the caller's constant-fixing step
/// (`tmp[0] += k - val(lbnd, tmp)` in [`super::series::Series::integ`]) is
/// unaffected either way, since it always *adds* the correction rather than
/// overwriting index 0.
pub(crate) fn int_once<T: Float>(c: &[T]) -> Vec<T> {
    let n = c.len();
    let mut tmp = vec![T::zero(); n + 1];
    tmp[0] = c[0];
    tmp[1] = -c[0];
    for j in 1..n {
        tmp[j] = tmp[j] + c[j];
        tmp[j + 1] = -c[j];
    }
    tmp
}

pub(crate) fn companion<T: Float + Debug>(c: &[T]) -> Result<Array<T>> {
    let len = c.len();
    if len == 2 {
        let v = T::one() + c[0] / c[1];
        return Array::from_vec_shape(vec![v], &[1, 1]);
    }
    let n = len - 1;
    let mut mat = vec![T::zero(); n * n];
    for k in 0..n {
        mat[k * n + k] = T::from(2 * k + 1).expect("index should convert to float type");
    }
    for k in 0..n - 1 {
        let v = -T::from(k + 1).expect("index should convert to float type");
        mat[k * n + (k + 1)] = v;
        mat[(k + 1) * n + k] = v;
    }
    let nt = T::from(n).expect("n should convert to float type");
    for i in 0..n {
        mat[i * n + (n - 1)] = mat[i * n + (n - 1)] + (c[i] / c[n]) * nt;
    }
    Array::from_vec_shape(mat, &[n, n])
}

pub(crate) fn to_power<T: Float>(c: &[T]) -> Vec<T> {
    let n = c.len();
    if n == 1 {
        return c.to_vec();
    }
    let mut c0 = vec![c[n - 2]];
    let mut c1 = vec![c[n - 1]];
    for i in (2..=n - 1).rev() {
        let it = T::from(i).expect("index should convert to float type");
        let tmp = c0;
        let scaled_c1: Vec<T> = c1
            .iter()
            .map(|&v| v * T::from(i - 1).expect("index should convert to float type") / it)
            .collect();
        c0 = sub_coefs(&[c[i - 2]], &scaled_c1);
        let scale_2i1 = T::from(2 * i - 1).expect("index should convert to float type");
        let scaled: Vec<T> = c1.iter().map(|&v| v * scale_2i1).collect();
        let mx = mulx_power(&c1);
        let diff = sub_coefs(&scaled, &mx);
        let diff_scaled: Vec<T> = diff.iter().map(|&v| v / it).collect();
        c1 = add_coefs(&tmp, &diff_scaled);
    }
    let mx = mulx_power(&c1);
    let diff = sub_coefs(&c1, &mx);
    add_coefs(&c0, &diff)
}

pub(crate) fn from_power<T: Float>(p: &[T]) -> Vec<T> {
    let mut res = vec![T::zero()];
    for i in (0..p.len()).rev() {
        res = mulx(&res);
        res = add_coefs(&res, &[p[i]]);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::super::{Laguerre, Polynomial};
    use approx::assert_relative_eq;

    #[test]
    fn eval_matches_hand_worked_l2() {
        // L_2(x) = (x^2-4x+2)/2; coef=[0,0,1] selects pure L_2. Pinned via
        // numpy: lagval([0,1,2],[0,0,1]) == [1., -0.5, -1.]
        let l2 = Laguerre::<f64>::from_coef(vec![0.0, 0.0, 1.0]);
        assert_relative_eq!(l2.eval(0.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(l2.eval(1.0), -0.5, epsilon = 1e-12);
        assert_relative_eq!(l2.eval(2.0), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn default_domain_and_window_are_zero_one() {
        let l = Laguerre::<f64>::from_coef(vec![1.0]);
        assert_relative_eq!(l.domain()[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(l.domain()[1], 1.0, epsilon = 1e-14);
        assert_relative_eq!(l.window()[0], 0.0, epsilon = 1e-14);
        assert_relative_eq!(l.window()[1], 1.0, epsilon = 1e-14);
    }

    #[test]
    fn deriv_matches_numpy_lagder_example() {
        // numpy: lagder((1,2,3,4)) == [-9, -7, -4]
        let p = Laguerre::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let d = p.deriv(1);
        let expected = [-9.0, -7.0, -4.0];
        for (got, want) in d.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-10);
        }
    }

    #[test]
    fn deriv_integ_round_trip() {
        let p = Laguerre::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let round_tripped = p.integ(1, &[]).deriv(1);
        for &x in &[0.05, 0.3, 0.6, 0.95] {
            assert_relative_eq!(round_tripped.eval(x), p.eval(x), epsilon = 1e-9);
        }
    }

    #[test]
    fn integ_with_nonzero_constant_matches_numpy_lagint_example() {
        // numpy: lagint((1,2,3,4), m=1, k=[5]) == [6, 1, 1, 1, -4]. Laguerre's
        // own `int_once` (unlike every other family here) leaves index 0 at
        // `c[0]` rather than zero, so this specifically exercises that the
        // shared `Series::integ` constant-fixing step still adds the
        // correction on top of that nonzero placeholder correctly.
        let p = Laguerre::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let integral = p.integ(1, &[5.0]);
        let expected = [6.0, 1.0, 1.0, 1.0, -4.0];
        assert_eq!(integral.coef().len(), expected.len());
        for (got, want) in integral.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-9);
        }
    }

    #[test]
    #[cfg(feature = "lapack")]
    fn fit_recovers_known_coefficients_exactly_noise_free() {
        // f(x) = 1*L0 + 2*L1 + 3*L2 = 1 + 2*(1-x) + 3*(x^2-4x+2)/2, x in
        // [0, 1] (Laguerre's own default domain/window, so the auto-fit
        // domain [min(x),max(x)]=[0,1] keeps the identity map).
        let n = 9;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64) / ((n - 1) as f64)).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 1.0 + 2.0 * (1.0 - x) + 3.0 * (x * x - 4.0 * x + 2.0) / 2.0)
            .collect();
        let x = crate::array::Array::from_vec(xs);
        let y = crate::array::Array::from_vec(ys);

        let fitted = Laguerre::<f64>::fit(&x, &y, 2).expect("noise-free fit should succeed");
        assert_relative_eq!(fitted.domain()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.domain()[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.coef()[0], 1.0, epsilon = 1e-7);
        assert_relative_eq!(fitted.coef()[1], 2.0, epsilon = 1e-7);
        assert_relative_eq!(fitted.coef()[2], 3.0, epsilon = 1e-7);
    }

    #[test]
    #[cfg(feature = "lapack")]
    fn roots_of_known_laguerre_series() {
        // L_2(x) roots are 2 +/- sqrt(2) (numpy: lagroots([0,0,1]) ==
        // [0.58578644, 3.41421356]).
        let p = Laguerre::<f64>::from_coef(vec![0.0, 0.0, 1.0]);
        let mut vals: Vec<f64> = p
            .roots()
            .expect("roots should succeed")
            .to_vec()
            .into_iter()
            .map(|z| z.re)
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs among real roots"));
        assert_eq!(vals.len(), 2);
        assert_relative_eq!(vals[0], 2.0 - 2.0_f64.sqrt(), epsilon = 1e-7);
        assert_relative_eq!(vals[1], 2.0 + 2.0_f64.sqrt(), epsilon = 1e-7);
    }

    #[test]
    fn convert_laguerre_to_power_matches_numpy_lag2poly_example() {
        // numpy: lag2poly((1,2,3,4)) == [10., -20., 7.5, -0.66666667]
        let l = Laguerre::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let p: Polynomial<f64> = l.convert();
        let expected = [10.0, -20.0, 7.5, -2.0 / 3.0];
        for (got, want) in p.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-8);
        }
    }

    #[test]
    fn convert_power_to_laguerre_matches_numpy_poly2lag_example() {
        // numpy: poly2lag((1,2,3,4)) == [33, -86, 78, -24]
        let p = Polynomial::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let l: Laguerre<f64> = p.convert();
        let expected = [33.0, -86.0, 78.0, -24.0];
        for (got, want) in l.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-8);
        }
    }

    #[test]
    fn mul_via_power_roundtrip_matches_numpy_lagmul_example() {
        // numpy: lagmul((1,2,3),(3,2,1)) == [10, 4, 16, -12, 18]
        let a = Laguerre::<f64>::from_coef(vec![1.0, 2.0, 3.0]);
        let b = Laguerre::<f64>::from_coef(vec![3.0, 2.0, 1.0]);
        let prod = a
            .mul(&b)
            .expect("same-domain multiplication should succeed");
        let expected = [10.0, 4.0, 16.0, -12.0, 18.0];
        assert_eq!(prod.coef().len(), expected.len());
        for (got, want) in prod.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-7);
        }
    }
}
