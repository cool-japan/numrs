//! Basis hooks for [`super::Hermite`] (`numpy.polynomial.Hermite` parity):
//! physicists' Hermite polynomials `H_i(x)`.
//!
//! Ported directly from `numpy.polynomial.hermite` (`hermval`, `hermvander`,
//! `hermmulx`, `hermder`, `hermint`, `hermcompanion`, `herm2poly`,
//! `poly2herm`), applying the recurrence `H_0=1, H_1=2x,
//! H_{n+1}=2x*H_n - 2n*H_{n-1}` numerically -- see [`super::chebyshev`]'s
//! module doc comment for why this (rather than routing through
//! `special::OrthogonalPolynomials::hermite`'s symbolic power-basis
//! generator) is the reuse this task called for.

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
    let mut nd = n;
    let mut c0 = c[n - 2];
    let mut c1 = c[n - 1];
    for i in 3..=n {
        let tmp = c0;
        nd -= 1;
        c0 = c[n - i] - c1 * T::from(2 * (nd - 1)).expect("index should convert to float type");
        c1 = tmp + c1 * x2;
    }
    c0 + c1 * x2
}

pub(crate) fn vander_row<T: Float>(x: T, deg: usize) -> Vec<T> {
    let mut v = vec![T::zero(); deg + 1];
    v[0] = T::one();
    if deg > 0 {
        let x2 = x + x;
        v[1] = x2;
        for i in 2..=deg {
            v[i] = v[i - 1] * x2
                - v[i - 2] * T::from(2 * (i - 1)).expect("index should convert to float type");
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
    let half = T::from(0.5).expect("0.5 should convert to float type");
    prd[1] = c[0] * half;
    for i in 1..n {
        prd[i + 1] = c[i] * half;
        prd[i - 1] = prd[i - 1] + c[i] * T::from(i).expect("index should convert to float type");
    }
    prd
}

/// One derivative step. Callers guarantee `c.len() >= 2`. Unlike Chebyshev
/// and Legendre, Hermite's derivative recurrence needs no cross-term
/// adjustment and never mutates its input.
pub(crate) fn der_once<T: Float>(c: &[T]) -> Vec<T> {
    let old_n = c.len();
    let new_n = old_n - 1;
    let mut der = vec![T::zero(); new_n];
    for j in (1..=new_n).rev() {
        der[j - 1] = T::from(2 * j).expect("index should convert to float type") * c[j];
    }
    der
}

/// One antiderivative step, ignoring the integration constant (index 0 is
/// left at zero here). The caller fixes it up via [`val`].
pub(crate) fn int_once<T: Float>(c: &[T]) -> Vec<T> {
    let n = c.len();
    let mut tmp = vec![T::zero(); n + 1];
    tmp[1] = c[0] / T::from(2.0).expect("2.0 should convert to float type");
    for j in 1..n {
        tmp[j + 1] = c[j] / T::from(2 * (j + 1)).expect("index should convert to float type");
    }
    tmp
}

pub(crate) fn companion<T: Float + Debug>(c: &[T]) -> Result<Array<T>> {
    let len = c.len();
    if len == 2 {
        let v = T::from(-0.5).expect("-0.5 should convert to float type") * c[0] / c[1];
        return Array::from_vec_shape(vec![v], &[1, 1]);
    }
    let n = len - 1;

    // scl = cumulative-product(1, 1/sqrt(2*(n-1)), 1/sqrt(2*(n-2)), ..., 1/sqrt(2))
    //       then reversed.
    let mut raw = vec![T::one(); n];
    for (k, raw_k) in raw.iter_mut().enumerate().take(n).skip(1) {
        let arg = T::from(2 * (n - k)).expect("index should convert to float type");
        *raw_k = T::one() / arg.sqrt();
    }
    let mut cum = vec![T::zero(); n];
    let mut running = T::one();
    for (k, &r) in raw.iter().enumerate() {
        running = running * r;
        cum[k] = running;
    }
    let mut scl = vec![T::zero(); n];
    for (k, scl_k) in scl.iter_mut().enumerate() {
        *scl_k = cum[n - 1 - k];
    }

    let half = T::from(0.5).expect("0.5 should convert to float type");
    let mut mat = vec![T::zero(); n * n];
    for k in 0..n - 1 {
        let v = (half * T::from(k + 1).expect("index should convert to float type")).sqrt();
        mat[k * n + (k + 1)] = v;
        mat[(k + 1) * n + k] = v;
    }
    let two = T::from(2.0).expect("2.0 should convert to float type");
    for i in 0..n {
        mat[i * n + (n - 1)] = mat[i * n + (n - 1)] - scl[i] * c[i] / (two * c[n]);
    }
    Array::from_vec_shape(mat, &[n, n])
}

pub(crate) fn to_power<T: Float>(c: &[T]) -> Vec<T> {
    let n = c.len();
    if n == 1 {
        return c.to_vec();
    }
    let two = T::from(2.0).expect("2.0 should convert to float type");
    if n == 2 {
        return vec![c[0], c[1] * two];
    }
    let mut c0 = vec![c[n - 2]];
    let mut c1 = vec![c[n - 1]];
    for i in (2..=n - 1).rev() {
        let tmp = c0;
        let scale = two * T::from(i - 1).expect("index should convert to float type");
        let scaled_c1: Vec<T> = c1.iter().map(|&v| v * scale).collect();
        c0 = sub_coefs(&[c[i - 2]], &scaled_c1);
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
    use super::super::{Hermite, Polynomial};
    use approx::assert_relative_eq;

    #[test]
    fn eval_matches_hand_worked_h2() {
        // H_2(x) = 4x^2 - 2; coef=[0,0,1] selects pure H_2. Pinned via
        // numpy: hermval([0,1,0.5],[0,0,1]) == [-2., 2., -1.]
        let h2 = Hermite::<f64>::from_coef(vec![0.0, 0.0, 1.0]);
        assert_relative_eq!(h2.eval(0.0), -2.0, epsilon = 1e-12);
        assert_relative_eq!(h2.eval(1.0), 2.0, epsilon = 1e-12);
        assert_relative_eq!(h2.eval(0.5), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn deriv_matches_numpy_hermder_example() {
        // numpy: hermder((1,2,3,4)) == [4, 12, 24]
        let p = Hermite::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let d = p.deriv(1);
        let expected = [4.0, 12.0, 24.0];
        for (got, want) in d.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-10);
        }
    }

    #[test]
    fn deriv_integ_round_trip() {
        let p = Hermite::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let round_tripped = p.integ(1, &[]).deriv(1);
        for &x in &[-0.8, -0.2, 0.4, 0.9] {
            assert_relative_eq!(round_tripped.eval(x), p.eval(x), epsilon = 1e-8);
        }
    }

    #[test]
    #[cfg(feature = "lapack")]
    fn fit_recovers_known_coefficients_exactly_noise_free() {
        // f(x) = 1*H0 + 2*H1 + 3*H2 = 1 + 4x + 3*(4x^2-2) = 12x^2 + 4x - 5
        let n = 9;
        let xs: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f64) / ((n - 1) as f64))
            .collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 1.0 + 2.0 * (2.0 * x) + 3.0 * (4.0 * x * x - 2.0))
            .collect();
        let x = crate::array::Array::from_vec(xs);
        let y = crate::array::Array::from_vec(ys);

        let fitted = Hermite::<f64>::fit(&x, &y, 2).expect("noise-free fit should succeed");
        assert_relative_eq!(fitted.coef()[0], 1.0, epsilon = 1e-7);
        assert_relative_eq!(fitted.coef()[1], 2.0, epsilon = 1e-7);
        assert_relative_eq!(fitted.coef()[2], 3.0, epsilon = 1e-7);
    }

    #[test]
    #[cfg(feature = "lapack")]
    fn roots_of_known_hermite_series() {
        // H_3(x) = 8x^3 - 12x, roots 0, +/- sqrt(3/2).
        // Pinned via numpy: hermroots([0,0,0,1]) == [-1.22474487, ~0, 1.22474487]
        let p = Hermite::<f64>::from_coef(vec![0.0, 0.0, 0.0, 1.0]);
        let mut vals: Vec<f64> = p
            .roots()
            .expect("roots should succeed")
            .to_vec()
            .into_iter()
            .map(|z| z.re)
            .collect();
        vals.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs among real roots"));
        assert_eq!(vals.len(), 3);
        assert_relative_eq!(vals[0], -1.224744871391589, epsilon = 1e-7);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-7);
        assert_relative_eq!(vals[2], 1.224744871391589, epsilon = 1e-7);
    }

    #[test]
    fn convert_hermite_to_power_matches_numpy_herm2poly_example() {
        // numpy: herm2poly((1,2,3,4)) == [-5., -44., 12., 32.]
        let h = Hermite::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let p: Polynomial<f64> = h.convert();
        let expected = [-5.0, -44.0, 12.0, 32.0];
        for (got, want) in p.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-8);
        }
    }

    #[test]
    fn convert_power_to_hermite_matches_numpy_poly2herm_example() {
        // numpy: poly2herm((1,2,3,4)) == [2.5, 4., 0.75, 0.5]
        let p = Polynomial::<f64>::from_coef(vec![1.0, 2.0, 3.0, 4.0]);
        let h: Hermite<f64> = p.convert();
        let expected = [2.5, 4.0, 0.75, 0.5];
        for (got, want) in h.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-10);
        }
    }

    #[test]
    fn mul_via_power_roundtrip_matches_numpy_hermmul_example() {
        // numpy: hermmul((1,2,3),(3,2,1)) == [35, 40, 38, 8, 3]
        let a = Hermite::<f64>::from_coef(vec![1.0, 2.0, 3.0]);
        let b = Hermite::<f64>::from_coef(vec![3.0, 2.0, 1.0]);
        let prod = a
            .mul(&b)
            .expect("same-domain multiplication should succeed");
        let expected = [35.0, 40.0, 38.0, 8.0, 3.0];
        assert_eq!(prod.coef().len(), expected.len());
        for (got, want) in prod.coef().iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-7);
        }
    }
}
