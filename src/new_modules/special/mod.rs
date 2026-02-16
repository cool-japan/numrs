//! Special mathematical functions implementation for NumRS2
//!
//! This module provides implementations of various special functions from mathematical physics
//! and engineering, similar to those found in the scipy.special library in Python.
//!
//! # Submodules
//!
//! - [`common`] - Common traits and types
//! - [`error_functions`] - Error functions (erf, erfc, erfinv, erfcinv)
//! - [`mod@gamma`] - Gamma and beta functions
//! - [`bessel`] - Bessel functions (J, Y, I, K)
//! - [`elliptic`] - Elliptic functions and integrals
//! - [`hypergeometric`] - Hypergeometric-related functions (zeta, Lambert W, polylog, etc.)
//! - [`orthogonal`] - Orthogonal polynomials and related functions (Legendre, Airy, Struve)

pub mod bessel;
pub mod common;
pub mod elliptic;
pub mod error_functions;
pub mod gamma;
pub mod hypergeometric;
pub mod orthogonal;

// Re-export all public functions for convenience

// Error functions
pub use error_functions::{erf, erfc, erfcinv, erfinv};

// Gamma and beta functions
pub use gamma::{beta, betainc, digamma, gamma, gammainc, gammaln};

// Bessel functions
pub use bessel::{bessel_i, bessel_j, bessel_k, bessel_y};

// Elliptic functions
pub use elliptic::{ellipe, ellipeinc, ellipf, ellipk, jacobi_elliptic};

// Hypergeometric and related functions
pub use hypergeometric::{exp1, expi, fresnel, lambertw, lambertwm1, polylog, shichi, sici, zeta};

// Orthogonal polynomials and related functions
pub use orthogonal::{
    airy_ai, airy_bi, associated_legendre_p, legendre_p, spherical_harmonic, struve_h,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use approx::assert_relative_eq;

    #[test]
    fn test_erf() {
        let values = Array::from_vec(vec![0.0, 0.5, 1.0, -0.5]);
        let result = erf(&values);

        // Known values of erf
        assert_relative_eq!(result.to_vec()[0], 0.0, epsilon = 1e-8);
        assert_relative_eq!(result.to_vec()[1], 0.5204998778130465, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[2], 0.8427007929497149, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[3], -0.5204998778130465, epsilon = 1e-4);
    }

    #[test]
    fn test_erfc() {
        let values = Array::from_vec(vec![0.0, 0.5, 1.0, 2.0]);
        let result = erfc(&values);

        // Known values of erfc
        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(result.to_vec()[1], 0.4795001221869535, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[2], 0.15729920705028513, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[3], 0.0046777349810472645, epsilon = 1e-4);
    }

    #[test]
    fn test_gamma() {
        let values = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = gamma(&values);

        // Known values of gamma
        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[3], 6.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[4], 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bessel_j() {
        let values = Array::from_vec(vec![0.0, 1.0, 2.0, 5.0]);

        // Bessel J0
        let j0 = bessel_j(0, &values);
        assert_relative_eq!(j0.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(j0.to_vec()[1], 0.7651976865579666, epsilon = 1e-4);

        // Bessel J1
        let j1 = bessel_j(1, &values);
        assert_relative_eq!(j1.to_vec()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(j1.to_vec()[1], 0.44005058574493355, epsilon = 1e-4);
    }

    #[test]
    fn test_ellipk() {
        let values = Array::from_vec(vec![0.0, 0.5, 0.9]);
        let result = ellipk(&values);

        // Known values
        assert_relative_eq!(
            result.to_vec()[0],
            std::f64::consts::PI / 2.0,
            epsilon = 1e-10
        );
        assert!(result.to_vec()[1] > std::f64::consts::PI / 2.0);
        assert!(result.to_vec()[2] > result.to_vec()[1]);
    }

    #[test]
    fn test_beta() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let result = beta(&a, &b).expect("beta should succeed");

        // B(1,1) = 1, B(2,2) = 1/6, B(3,3) = 1/30
        assert_relative_eq!(result.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[1], 1.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(result.to_vec()[2], 1.0 / 30.0, epsilon = 1e-10);
    }

    #[test]
    fn test_betainc() {
        let a = Array::from_vec(vec![1.0, 2.0]);
        let b = Array::from_vec(vec![1.0, 2.0]);
        let x = Array::from_vec(vec![0.5, 0.5]);
        let result = betainc(&a, &b, &x).expect("betainc should succeed");

        // I_0.5(1,1) = 0.5, I_0.5(2,2) = 0.5
        assert_relative_eq!(result.to_vec()[0], 0.5, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[1], 0.5, epsilon = 1e-4);
    }

    #[test]
    fn test_zeta() {
        let s = Array::from_vec(vec![2.0, 3.0, 4.0]);
        let result = zeta(&s);

        // zeta(2) = pi^2/6 ~ 1.6449, zeta(3) ~ 1.2021, zeta(4) = pi^4/90 ~ 1.0823
        assert_relative_eq!(
            result.to_vec()[0],
            std::f64::consts::PI.powi(2) / 6.0,
            epsilon = 1e-3
        );
        assert_relative_eq!(result.to_vec()[1], 1.2020569, epsilon = 1e-3);
        assert_relative_eq!(
            result.to_vec()[2],
            std::f64::consts::PI.powi(4) / 90.0,
            epsilon = 1e-3
        );
    }

    #[test]
    fn test_airy_ai() {
        let x = Array::from_vec(vec![0.0, 1.0, -1.0]);
        let result = airy_ai(&x);

        // Ai(0) ~ 0.35503, Ai(1) ~ 0.13529, Ai(-1) ~ 0.53556
        assert_relative_eq!(result.to_vec()[0], 0.35502805388781724, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[1], 0.13529241631288141, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[2], 0.5355608832923521, epsilon = 1e-4);
    }

    #[test]
    fn test_airy_bi() {
        let x = Array::from_vec(vec![0.0, 1.0, -1.0]);
        let result = airy_bi(&x);

        // Bi(0) ~ 0.61493, Bi(1) ~ 1.20743, Bi(-1) ~ 0.10399
        assert_relative_eq!(result.to_vec()[0], 0.61492662744600073, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[1], 1.2074283264132947, epsilon = 1e-4);
        assert_relative_eq!(result.to_vec()[2], 0.10399738949694461, epsilon = 1e-4);
    }

    #[test]
    fn test_sici() {
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let (si, ci) = sici(&x);

        // Si(0) = 0, Si(1) ~ 0.9461, Si(2) ~ 1.6054
        assert_relative_eq!(si.to_vec()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(si.to_vec()[1], 0.9460830703671830, epsilon = 1e-3);
        assert_relative_eq!(si.to_vec()[2], 1.6054129768026948, epsilon = 1e-3);

        // Ci(1) ~ 0.3374, Ci(2) ~ 0.4230
        assert_relative_eq!(ci.to_vec()[1], 0.33740392290096813, epsilon = 1e-3);
        assert_relative_eq!(ci.to_vec()[2], 0.4229808287748649, epsilon = 1e-3);
    }

    #[test]
    fn test_fresnel() {
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let (s, c) = fresnel(&x);

        // S(0) = 0, C(0) = 0
        assert_relative_eq!(s.to_vec()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(c.to_vec()[0], 0.0, epsilon = 1e-10);

        // S(1) ~ 0.4383, C(1) ~ 0.7799
        assert_relative_eq!(s.to_vec()[1], 0.43825914739035476, epsilon = 1e-3);
        assert_relative_eq!(c.to_vec()[1], 0.7798934003768228, epsilon = 1e-3);
    }

    #[test]
    fn test_expi() {
        let x = Array::from_vec(vec![1.0, 2.0]);
        let result = expi(&x);

        // Ei(1) ~ 1.8951, Ei(2) ~ 4.9542
        assert_relative_eq!(result.to_vec()[0], 1.8951178163559368, epsilon = 1e-3);
        assert_relative_eq!(result.to_vec()[1], 4.954234356001890, epsilon = 1e-3);
    }

    #[test]
    fn test_exp1() {
        let x = Array::from_vec(vec![1.0, 2.0]);
        let result = exp1(&x);

        // E1(1) ~ 0.2194, E1(2) ~ 0.0489
        assert_relative_eq!(result.to_vec()[0], 0.21938393439552027, epsilon = 1e-3);
        assert_relative_eq!(result.to_vec()[1], 0.04890051070806112, epsilon = 1e-3);
    }

    #[test]
    fn test_legendre_p() {
        let x = Array::from_vec(vec![0.0, 0.5, 1.0]);

        // P_0(x) = 1
        let p0 = legendre_p(0, &x).expect("legendre_p should succeed");
        assert_relative_eq!(p0.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(p0.to_vec()[1], 1.0, epsilon = 1e-10);
        assert_relative_eq!(p0.to_vec()[2], 1.0, epsilon = 1e-10);

        // P_1(x) = x
        let p1 = legendre_p(1, &x).expect("legendre_p should succeed");
        assert_relative_eq!(p1.to_vec()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(p1.to_vec()[1], 0.5, epsilon = 1e-10);
        assert_relative_eq!(p1.to_vec()[2], 1.0, epsilon = 1e-10);

        // P_2(x) = (3x^2 - 1)/2
        let p2 = legendre_p(2, &x).expect("legendre_p should succeed");
        assert_relative_eq!(p2.to_vec()[0], -0.5, epsilon = 1e-10); // P_2(0) = -0.5
        assert_relative_eq!(p2.to_vec()[1], -0.125, epsilon = 1e-10); // P_2(0.5) = -0.125
        assert_relative_eq!(p2.to_vec()[2], 1.0, epsilon = 1e-10); // P_2(1) = 1
    }

    #[test]
    fn test_associated_legendre_p() {
        let x = Array::from_vec(vec![0.0, 0.5, 1.0]);

        // P_1^1(x) = -sqrt(1-x^2)
        let p11 = associated_legendre_p(1, 1, &x).expect("associated_legendre_p should succeed");
        assert_relative_eq!(p11.to_vec()[0], -1.0, epsilon = 1e-10); // P_1^1(0) = -1
        assert_relative_eq!(p11.to_vec()[1], -0.8660254037844386, epsilon = 1e-10); // P_1^1(0.5)
        assert_relative_eq!(p11.to_vec()[2], 0.0, epsilon = 1e-10); // P_1^1(1) = 0
    }

    #[test]
    fn test_spherical_harmonic() {
        let theta = Array::from_vec(vec![std::f64::consts::PI / 2.0]);
        let phi = Array::from_vec(vec![0.0]);

        // Y_0^0 = 1/(2*sqrt(pi)) ~ 0.2821
        let y00 =
            spherical_harmonic(0, 0, &theta, &phi).expect("spherical_harmonic should succeed");
        assert_relative_eq!(y00.to_vec()[0], 0.2820947917738782, epsilon = 1e-4);
    }

    #[test]
    fn test_jacobi_elliptic_special_cases() {
        let u = Array::from_vec(vec![0.0, 1.0, 2.0]);

        // m = 0: sn(u, 0) = sin(u), cn(u, 0) = cos(u), dn(u, 0) = 1
        let (sn, cn, dn) = jacobi_elliptic(&u, 0.0).expect("jacobi_elliptic should succeed");
        assert_relative_eq!(sn.to_vec()[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(cn.to_vec()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(dn.to_vec()[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_incomplete_elliptic_integrals() {
        // F(0, m) = 0 for any m
        let phi = Array::from_vec(vec![0.0]);
        let m = Array::from_vec(vec![0.5]);
        let f = ellipf(&phi, &m).expect("ellipf should succeed");
        assert_relative_eq!(f.to_vec()[0], 0.0, epsilon = 1e-10);

        // E(0, m) = 0 for any m
        let e = ellipeinc(&phi, &m).expect("ellipeinc should succeed");
        assert_relative_eq!(e.to_vec()[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lambertw() {
        let x = Array::from_vec(vec![0.0, 1.0]);
        let result = lambertw(&x);

        // W(0) = 0
        assert_relative_eq!(result.to_vec()[0], 0.0, epsilon = 1e-10);

        // W(1) ~ 0.5671 (Omega constant)
        assert_relative_eq!(result.to_vec()[1], 0.5671432904097839, epsilon = 1e-6);
    }

    #[test]
    fn test_lambertw_identity() {
        // Verify W(x) * e^(W(x)) = x
        let x = Array::from_vec(vec![0.5, 1.0, 2.0, 5.0]);
        let w = lambertw(&x);

        for i in 0..x.to_vec().len() {
            let x_val: f64 = x.to_vec()[i];
            let w_val: f64 = w.to_vec()[i];
            let reconstructed = w_val * w_val.exp();
            assert_relative_eq!(reconstructed, x_val, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_polylog() {
        // Li_2(0.5) ~ 0.5822
        let z = Array::from_vec(vec![0.5]);
        let result = polylog(2.0, &z);
        assert_relative_eq!(result.to_vec()[0], 0.5822405264650125, epsilon = 1e-3);
    }

    #[test]
    fn test_struve_h() {
        // H_0(0) = 0
        let x = Array::from_vec(vec![0.0, 1.0]);
        let result = struve_h(0, &x);
        assert_relative_eq!(result.to_vec()[0], 0.0, epsilon = 1e-10);
        // H_0(1) - test that it's in reasonable range
        assert!(result.to_vec()[1] > 0.3 && result.to_vec()[1] < 0.7);
    }
}
