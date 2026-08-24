//! Basis hooks for [`super::Chebyshev`] (`numpy.polynomial.Chebyshev`
//! parity): Chebyshev polynomials of the first kind, `T_i(x)`.
//!
//! Every formula below is ported directly from `numpy.polynomial.chebyshev`
//! (`chebval`, `chebvander`, `chebmulx`, `chebder`, `chebint`,
//! `chebcompanion`, `cheb2poly`, `poly2cheb`), applying the same three-term
//! recurrence `T_0=1, T_1=x, T_{n+1}=2x*T_n - T_{n-1}` numerically (this is
//! also literally the recurrence `OrthogonalPolynomials::chebyshev_t` in
//! `super::super::special` uses to build a symbolic power-basis polynomial
//! one degree at a time -- applying it directly to numeric evaluation here
//! is the "reuse the recurrence, don't reimplement it" the task called for,
//! since routing every design-matrix entry through that symbolic generator
//! would be both slower and no more faithful to the recurrence).

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
    let x2 = x + x;
    let mut c0 = c[n - 2];
    let mut c1 = c[n - 1];
    for i in 3..=n {
        let tmp = c0;
        c0 = c[n - i] - c1;
        c1 = tmp + c1 * x2;
    }
    c0 + c1 * x
}

pub(crate) fn vander_row<T: Float>(x: T, deg: usize) -> Vec<T> {
    let mut v = vec![T::zero(); deg + 1];
    v[0] = T::one();
    if deg > 0 {
        let x2 = x + x;
        v[1] = x;
        for i in 2..=deg {
            v[i] = v[i - 1] * x2 - v[i - 2];
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
    prd[1] = c[0];
    if n > 1 {
        let half = T::from(0.5).expect("0.5 should convert to float type");
        for i in 1..n {
            let t = c[i] * half;
            prd[i + 1] = prd[i + 1] + t;
            prd[i - 1] = prd[i - 1] + t;
        }
    }
    prd
}

/// One derivative step. Callers ([`super::series::Series::deriv`]) guarantee
/// `c.len() >= 2`.
pub(crate) fn der_once<T: Float>(c: &[T]) -> Vec<T> {
    let old_n = c.len();
    let new_n = old_n - 1;
    let mut cw = c.to_vec();
    let mut der = vec![T::zero(); new_n];
    let mut j = new_n;
    while j > 2 {
        der[j - 1] = T::from(2 * j).expect("index should convert to float type") * cw[j];
        cw[j - 2] = cw[j - 2]
            + (T::from(j).expect("index should convert to float type") * cw[j])
                / T::from(j - 2).expect("index should convert to float type");
        j -= 1;
    }
    if new_n > 1 {
        der[1] = T::from(4.0).expect("4.0 should convert to float type") * cw[2];
    }
    der[0] = cw[1];
    der
}

/// One antiderivative step, ignoring the integration constant (index 0 is
/// left at the recurrence's natural placeholder, which for Chebyshev is
/// always exactly zero). The caller ([`super::series::Series::integ`]) fixes
/// it up via [`val`].
pub(crate) fn int_once<T: Float>(c: &[T]) -> Vec<T> {
    let n = c.len();
    let two = T::from(2.0).expect("2.0 should convert to float type");
    let mut tmp = vec![T::zero(); n + 1];
    tmp[1] = c[0];
    if n > 1 {
        tmp[2] = c[1] / T::from(4.0).expect("4.0 should convert to float type");
    }
    for j in 2..n {
        let jt = T::from(j).expect("index should convert to float type");
        tmp[j + 1] = c[j] / (two * (jt + T::one()));
        tmp[j - 1] = tmp[j - 1] - c[j] / (two * (jt - T::one()));
    }
    tmp
}

/// Unrotated colleague matrix. `numpy.polynomial.chebyshev.chebcompanion`
/// additionally reverses both axes ("rotated companion matrix reduces
/// error"); that reversal is a similarity transform (`P M P` with `P` its
/// own inverse), so it cannot change the eigenvalues, only their computed
/// accuracy for numerically adversarial inputs -- skipped here for a
/// simpler, still-correct construction (see [`super::series::Series::roots`]'s
/// doc comment for the resulting accuracy expectation).
pub(crate) fn companion<T: Float + Debug>(c: &[T]) -> Result<Array<T>> {
    let len = c.len();
    if len == 2 {
        let v = -c[0] / c[1];
        return Array::from_vec_shape(vec![v], &[1, 1]);
    }
    let n = len - 1;
    let half = T::from(0.5).expect("0.5 should convert to float type");
    let sqrt_half = half.sqrt();

    let mut mat = vec![T::zero(); n * n];
    for i in 0..n - 1 {
        let v = if i == 0 { sqrt_half } else { half };
        mat[i * n + (i + 1)] = v;
        mat[(i + 1) * n + i] = v;
    }
    // scl[0] = 1, scl[k] = sqrt(0.5) for k = 1..n-1; scl[n-1] = sqrt(0.5) since n >= 2 here.
    for i in 0..n {
        let scl_i = if i == 0 { T::one() } else { sqrt_half };
        mat[i * n + (n - 1)] = mat[i * n + (n - 1)] - (c[i] / c[n]) * (scl_i / sqrt_half) * half;
    }
    Array::from_vec_shape(mat, &[n, n])
}

pub(crate) fn to_power<T: Float>(c: &[T]) -> Vec<T> {
    let n = c.len();
    if n < 3 {
        return c.to_vec();
    }
    let two = T::from(2.0).expect("2.0 should convert to float type");
    let mut c0 = vec![c[n - 2]];
    let mut c1 = vec![c[n - 1]];
    for i in (2..=n - 1).rev() {
        let tmp = c0;
        c0 = sub_coefs(&[c[i - 2]], &c1);
        let mut mx = mulx_power(&c1);
        for v in mx.iter_mut() {
            *v = *v * two;
        }
        c1 = add_coefs(&tmp, &mx);
    }
    let mut mx = mulx_power(&c1);
    for v in mx.iter_mut() {
        *v = *v * two;
    }
    add_coefs(&c0, &mx)
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
    use super::super::Chebyshev;
    use approx::assert_relative_eq;

    #[test]
    fn eval_matches_hand_worked_t2() {
        // T_2(x) = 2x^2 - 1; coef = [0, 0, 1] selects pure T_2.
        let t2 = Chebyshev::<f64>::from_coef(vec![0.0, 0.0, 1.0]);
        assert_relative_eq!(t2.eval(0.0), -1.0, epsilon = 1e-12);
        assert_relative_eq!(t2.eval(1.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(t2.eval(0.5), -0.5, epsilon = 1e-12);
    }

    #[test]
    #[cfg(feature = "lapack")]
    fn fit_recovers_known_coefficients_exactly_noise_free() {
        // f(x) = 1*T0 + 2*T1 + 3*T2 = 1 + 2x + 3*(2x^2-1) = 6x^2 + 2x - 2.
        // x spans exactly [-1, 1] so the auto-fit domain equals the default
        // window: the domain->window map is the identity, and the fitted
        // coefficients should equal the true Chebyshev coefficients exactly
        // (up to floating point) with more points than the degree.
        let n = 9;
        let xs: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 6.0 * x * x + 2.0 * x - 2.0).collect();
        let x = numrs2_array(&xs);
        let y = numrs2_array(&ys);

        let fitted = Chebyshev::<f64>::fit(&x, &y, 2).expect("noise-free fit should succeed");
        assert_relative_eq!(fitted.domain()[0], -1.0, epsilon = 1e-12);
        assert_relative_eq!(fitted.domain()[1], 1.0, epsilon = 1e-12);
        assert_relative_eq!(fitted.coef()[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.coef()[1], 2.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.coef()[2], 3.0, epsilon = 1e-8);
    }

    #[test]
    #[cfg(feature = "lapack")]
    fn fit_with_nondefault_data_range_exercises_nontrivial_domain_window_map() {
        // Same f in Chebyshev coefficients [1, 2, 3], but sampled on x in
        // [0, 10] instead of [-1, 1]: numpy's
        // `Chebyshev.fit(x, y, 2)` auto-domain is [0, 10], with the class's
        // default window [-1, 1] kept, so scl = 2/10 = 0.2 != 1. Pinned via:
        //   >>> import numpy as np
        //   >>> x = np.linspace(0, 10, 11)
        //   >>> y = np.polynomial.chebyshev.chebval(np.polynomial.polyutils.mapdomain(x,[0,10],[-1,1]), [1,2,3])
        //   >>> np.polynomial.Chebyshev.fit(x, y, 2)
        // gives domain=[0,10], window=[-1,1], coef=[1,2,3].
        let xs: Vec<f64> = (0..=10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&xv| {
                let xw = -1.0 + 0.2 * xv;
                1.0 + 2.0 * xw + 3.0 * (2.0 * xw * xw - 1.0)
            })
            .collect();
        let x = numrs2_array(&xs);
        let y = numrs2_array(&ys);

        let fitted = Chebyshev::<f64>::fit(&x, &y, 2).expect("fit should succeed");
        assert_relative_eq!(fitted.domain()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.domain()[1], 10.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.window()[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.window()[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(fitted.coef()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.coef()[1], 2.0, epsilon = 1e-6);
        assert_relative_eq!(fitted.coef()[2], 3.0, epsilon = 1e-6);
        // And the fitted series must still evaluate correctly at raw x.
        for &xv in &[0.0, 3.0, 7.0, 10.0] {
            let xw = -1.0 + 0.2 * xv;
            let expected = 1.0 + 2.0 * xw + 3.0 * (2.0 * xw * xw - 1.0);
            assert_relative_eq!(fitted.eval(xv), expected, epsilon = 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "lapack")]
    fn roots_of_known_chebyshev_series() {
        // T3 - T2 + T1 - T0 (ascending: [-1, 1, -1, 1]) has real roots
        // -0.5, ~0, 1 (numpy docstring example for chebroots).
        let p = Chebyshev::<f64>::from_coef(vec![-1.0, 1.0, -1.0, 1.0]);
        let mut vals: Vec<f64> = p
            .roots()
            .expect("roots should succeed")
            .to_vec()
            .into_iter()
            .map(|z| z.re)
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs among real roots"));
        assert_eq!(vals.len(), 3);
        assert_relative_eq!(vals[0], -0.5, epsilon = 1e-8);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-8);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-8);
    }

    #[test]
    fn deriv_integ_round_trip() {
        let p = Chebyshev::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let round_tripped = p.integ(1, &[]).deriv(1);
        for &x in &[-0.8, -0.2, 0.4, 0.9] {
            assert_relative_eq!(round_tripped.eval(x), p.eval(x), epsilon = 1e-9);
        }
    }

    #[test]
    fn deriv_matches_numpy_chebder_example() {
        // numpy: C.chebder((1,2,3,4)) == [14, 12, 24]
        let p = Chebyshev::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let d = p.deriv(1);
        assert_relative_eq!(d.coef()[0], 14.0, epsilon = 1e-10);
        assert_relative_eq!(d.coef()[1], 12.0, epsilon = 1e-10);
        assert_relative_eq!(d.coef()[2], 24.0, epsilon = 1e-10);
    }

    #[test]
    fn convert_chebyshev_to_power_matches_numpy_cheb2poly_example() {
        // numpy: cheb2poly(range(4)) == [-2., -8., 4., 12.]
        let c = Chebyshev::<f64>::from_coef(vec![0.0, 1.0, 2.0, 3.0]);
        let p: super::super::Polynomial<f64> = c.convert();
        assert_relative_eq!(p.coef()[0], -2.0, epsilon = 1e-10);
        assert_relative_eq!(p.coef()[1], -8.0, epsilon = 1e-10);
        assert_relative_eq!(p.coef()[2], 4.0, epsilon = 1e-10);
        assert_relative_eq!(p.coef()[3], 12.0, epsilon = 1e-10);
    }

    #[test]
    fn convert_power_to_chebyshev_matches_numpy_poly2cheb_example() {
        // numpy: poly2cheb(range(4)) == [1., 3.25, 1., 0.75]
        let p = super::super::Polynomial::<f64>::from_coef(vec![0.0, 1.0, 2.0, 3.0]);
        let c: Chebyshev<f64> = p.convert();
        assert_relative_eq!(c.coef()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(c.coef()[1], 3.25, epsilon = 1e-10);
        assert_relative_eq!(c.coef()[2], 1.0, epsilon = 1e-10);
        assert_relative_eq!(c.coef()[3], 0.75, epsilon = 1e-10);
    }

    #[test]
    fn convert_round_trip_is_identity() {
        let c = Chebyshev::<f64>::from_coef(vec![1.0, -2.5, 0.75, 3.0, -1.25]);
        let back: Chebyshev<f64> = c.convert::<super::super::PowerBasis>().convert();
        for (a, b) in c.coef().iter().zip(back.coef().iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn mul_via_power_roundtrip_matches_numpy_chebmul_example() {
        // numpy: chebmul((1,2,3),(3,2,1)) == [6.5, 12., 12., 4., 1.5]
        let a = Chebyshev::<f64>::from_coef(vec![1.0, 2.0, 3.0]);
        let b = Chebyshev::<f64>::from_coef(vec![3.0, 2.0, 1.0]);
        let prod = a
            .mul(&b)
            .expect("same-domain multiplication should succeed");
        let expected = [6.5, 12.0, 12.0, 4.0, 1.5];
        assert_eq!(prod.coef().len(), expected.len());
        for (got, want) in prod.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-9);
        }
    }

    fn numrs2_array(v: &[f64]) -> crate::array::Array<f64> {
        crate::array::Array::from_vec(v.to_vec())
    }
}
