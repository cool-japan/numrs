//! Basis hooks for [`super::Legendre`] (`numpy.polynomial.Legendre` parity):
//! Legendre polynomials `P_i(x)`.
//!
//! Ported directly from `numpy.polynomial.legendre` (`legval`, `legvander`,
//! `legmulx`, `legder`, `legint`, `legcompanion`, `leg2poly`, `poly2leg`),
//! applying the recurrence `(n+1)P_{n+1}(x) = (2n+1)x*P_n(x) - n*P_{n-1}(x)`
//! numerically -- see [`super::chebyshev`]'s module doc comment for why this
//! (rather than routing through `special::OrthogonalPolynomials::legendre`'s
//! symbolic power-basis generator) is the reuse this task called for.

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
        c0 = c[n - i] - c1 * (T::from(nd - 1).expect("index should convert to float type") / ndt);
        c1 =
            tmp + c1 * x * (T::from(2 * nd - 1).expect("index should convert to float type") / ndt);
    }
    c0 + c1 * x
}

pub(crate) fn vander_row<T: Float>(x: T, deg: usize) -> Vec<T> {
    let mut v = vec![T::zero(); deg + 1];
    v[0] = T::one();
    if deg > 0 {
        v[1] = x;
        for i in 2..=deg {
            let it = T::from(i).expect("index should convert to float type");
            v[i] =
                (v[i - 1] * x * (T::from(2 * i - 1).expect("index should convert to float type"))
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
    prd[1] = c[0];
    for i in 1..n {
        let j = i + 1;
        let k = i - 1;
        let s = i + j;
        let it = T::from(i).expect("index should convert to float type");
        let jt = T::from(j).expect("index should convert to float type");
        let st = T::from(s).expect("index should convert to float type");
        prd[j] = c[i] * jt / st;
        prd[k] = prd[k] + c[i] * it / st;
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
    while j > 2 {
        der[j - 1] = T::from(2 * j - 1).expect("index should convert to float type") * cw[j];
        cw[j - 2] = cw[j - 2] + cw[j];
        j -= 1;
    }
    if new_n > 1 {
        der[1] = T::from(3.0).expect("3.0 should convert to float type") * cw[2];
    }
    der[0] = cw[1];
    der
}

/// One antiderivative step, ignoring the integration constant (index 0 is
/// left at zero here, same as Chebyshev). The caller fixes it up via [`val`].
pub(crate) fn int_once<T: Float>(c: &[T]) -> Vec<T> {
    let n = c.len();
    let mut tmp = vec![T::zero(); n + 1];
    tmp[1] = c[0];
    if n > 1 {
        tmp[2] = c[1] / T::from(3.0).expect("3.0 should convert to float type");
    }
    for j in 2..n {
        let t = c[j] / T::from(2 * j + 1).expect("index should convert to float type");
        tmp[j + 1] = t;
        tmp[j - 1] = tmp[j - 1] - t;
    }
    tmp
}

pub(crate) fn companion<T: Float + Debug>(c: &[T]) -> Result<Array<T>> {
    let len = c.len();
    if len == 2 {
        let v = -c[0] / c[1];
        return Array::from_vec_shape(vec![v], &[1, 1]);
    }
    let n = len - 1;
    let mut scl = vec![T::zero(); n];
    for (k, scl_k) in scl.iter_mut().enumerate() {
        *scl_k = T::one()
            / T::from(2 * k + 1)
                .expect("index should convert to float type")
                .sqrt();
    }

    let mut mat = vec![T::zero(); n * n];
    for k in 0..n - 1 {
        let v = T::from(k + 1).expect("index should convert to float type") * scl[k] * scl[k + 1];
        mat[k * n + (k + 1)] = v;
        mat[(k + 1) * n + k] = v;
    }
    let n_over = T::from(n).expect("n should convert to float type")
        / T::from(2 * n - 1).expect("2n-1 should convert to float type");
    for i in 0..n {
        mat[i * n + (n - 1)] =
            mat[i * n + (n - 1)] - (c[i] / c[n]) * (scl[i] / scl[n - 1]) * n_over;
    }
    Array::from_vec_shape(mat, &[n, n])
}

pub(crate) fn to_power<T: Float>(c: &[T]) -> Vec<T> {
    let n = c.len();
    if n < 3 {
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
        let mx = mulx_power(&c1);
        let scaled_mx: Vec<T> = mx
            .iter()
            .map(|&v| v * T::from(2 * i - 1).expect("index should convert to float type") / it)
            .collect();
        c1 = add_coefs(&tmp, &scaled_mx);
    }
    add_coefs(&c0, &mulx_power(&c1))
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
    use super::super::{Legendre, Polynomial};
    use approx::assert_relative_eq;

    #[test]
    fn eval_matches_hand_worked_p2() {
        // P_2(x) = (3x^2-1)/2; coef=[0,0,1] selects pure P_2.
        let p2 = Legendre::<f64>::from_coef(vec![0.0, 0.0, 1.0]);
        assert_relative_eq!(p2.eval(0.0), -0.5, epsilon = 1e-12);
        assert_relative_eq!(p2.eval(1.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(p2.eval(0.5), -0.125, epsilon = 1e-12);
    }

    #[test]
    #[cfg(feature = "lapack")]
    fn fit_recovers_known_coefficients_exactly_noise_free() {
        // f(x) = 1*P0 + 2*P1 + 3*P2 = 1 + 2x + 3*(3x^2-1)/2
        let n = 9;
        let xs: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 1.0 + 2.0 * x + 3.0 * (3.0 * x * x - 1.0) / 2.0)
            .collect();
        let x = crate::array::Array::from_vec(xs);
        let y = crate::array::Array::from_vec(ys);

        let fitted = Legendre::<f64>::fit(&x, &y, 2).expect("noise-free fit should succeed");
        assert_relative_eq!(fitted.coef()[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.coef()[1], 2.0, epsilon = 1e-8);
        assert_relative_eq!(fitted.coef()[2], 3.0, epsilon = 1e-8);
    }

    #[test]
    #[cfg(feature = "lapack")]
    fn roots_of_known_legendre_series() {
        // P_2(x) = (3x^2-1)/2 has roots +/- 1/sqrt(3).
        let p = Legendre::<f64>::from_coef(vec![0.0, 0.0, 1.0]);
        let mut vals: Vec<f64> = p
            .roots()
            .expect("roots should succeed")
            .to_vec()
            .into_iter()
            .map(|z| z.re)
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs among real roots"));
        let expected = 1.0 / 3.0_f64.sqrt();
        assert_eq!(vals.len(), 2);
        assert_relative_eq!(vals[0], -expected, epsilon = 1e-8);
        assert_relative_eq!(vals[1], expected, epsilon = 1e-8);
    }

    #[test]
    fn deriv_integ_round_trip() {
        let p = Legendre::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let round_tripped = p.integ(1, &[]).deriv(1);
        for &x in &[-0.8, -0.2, 0.4, 0.9] {
            assert_relative_eq!(round_tripped.eval(x), p.eval(x), epsilon = 1e-9);
        }
    }

    #[test]
    fn convert_legendre_to_power_matches_numpy_leg2poly_example() {
        // numpy: from numpy.polynomial.legendre import leg2poly
        //        leg2poly([1,2,3,4]) == [-0.5, -4., 4.5, 10.]
        let l = Legendre::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let p: Polynomial<f64> = l.convert();
        let expected = [-0.5, -4.0, 4.5, 10.0];
        for (got, want) in p.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-9);
        }
    }

    #[test]
    fn convert_round_trip_is_identity() {
        let l = Legendre::<f64>::from_coef(vec![1.0, -2.5, 0.75, 3.0, -1.25]);
        let back: Legendre<f64> = l.convert::<super::super::PowerBasis>().convert();
        for (a, b) in l.coef().iter().zip(back.coef().iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }
}
