//! Special Mathematical Functions Module
//!
//! This module provides comprehensive special mathematical functions commonly used in
//! scientific computing, engineering, statistics, and physics. Built on top of `scirs2-special`,
//! it includes:
//!
//! - **Gamma Functions**: Gamma, log-gamma, beta, incomplete gamma
//! - **Error Functions**: erf, erfc, erfcx, erfi, inverse erf
//! - **Bessel Functions**: J (first kind), Y (second kind), I (modified first kind), K (modified second kind), spherical
//! - **Elliptic Integrals**: Complete and incomplete elliptic integrals of first and second kind
//! - **Orthogonal Polynomials**: Legendre, Chebyshev (T, U), Hermite, Laguerre, Jacobi
//! - **Airy Functions**: Ai, Bi and their derivatives
//! - **Hypergeometric Functions**: 1F1, 2F1
//! - **Spherical Harmonics**: Real and complex spherical harmonics
//! - **Zeta Functions**: Riemann and Hurwitz zeta functions
//! - **Mathieu Functions**: Solutions to Mathieu's differential equation
//!
//! # Examples
//!
//! ## Gamma and Beta Functions
//!
//! ```
//! use numrs2::special;
//!
//! // Gamma function: Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt
//! let gamma_4_5: f64 = special::gamma(4.5);
//! println!("Γ(4.5) = {}", gamma_4_5);
//!
//! // Log-gamma for numerical stability with large arguments
//! let log_gamma_100: f64 = special::loggamma(100.0);
//! println!("log Γ(100) = {}", log_gamma_100);
//!
//! // Beta function: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
//! let beta_2_3: f64 = special::beta(2.0, 3.0);
//! println!("B(2,3) = {}", beta_2_3);
//! ```
//!
//! ## Error Functions
//!
//! ```
//! use numrs2::special;
//!
//! // Error function: erf(x) = (2/√π) ∫₀^x e^(-t²) dt
//! let erf_1: f64 = special::erf(1.0);
//! println!("erf(1) = {}", erf_1);
//!
//! // Complementary error function: erfc(x) = 1 - erf(x)
//! let erfc_1: f64 = special::erfc(1.0);
//! println!("erfc(1) = {}", erfc_1);
//!
//! // Inverse error function
//! let inv_erf: f64 = special::erfinv(0.8);
//! println!("erf⁻¹(0.8) = {}", inv_erf);
//! ```
//!
//! ## Bessel Functions
//!
//! ```
//! use numrs2::special;
//!
//! // Bessel function of the first kind: J₀(x)
//! let j0_1: f64 = special::j0(1.0);
//! println!("J₀(1) = {}", j0_1);
//!
//! // Modified Bessel function of the first kind: I₀(x)
//! let i0_1: f64 = special::i0(1.0);
//! println!("I₀(1) = {}", i0_1);
//!
//! // Spherical Bessel function: j₀(x)
//! let sj0_1: f64 = special::spherical_jn(0, 1.0);
//! println!("j₀(1) = {}", sj0_1);
//! ```
//!
//! ## Elliptic Integrals
//!
//! ```
//! use numrs2::special;
//!
//! // Complete elliptic integral of the first kind: K(m)
//! let k1: f64 = special::ellipk(0.5);
//! println!("K(0.5) = {}", k1);
//!
//! // Complete elliptic integral of the second kind: E(m)
//! let e1: f64 = special::ellipe(0.5);
//! println!("E(0.5) = {}", e1);
//! ```
//!
//! ## Orthogonal Polynomials
//!
//! ```
//! use numrs2::special;
//!
//! // Legendre polynomial: P₃(0.5)
//! let p3: f64 = special::legendre(3, 0.5);
//! println!("P₃(0.5) = {}", p3);
//!
//! // Chebyshev polynomial of the first kind: T₃(0.5)
//! let t3: f64 = special::chebyshev(3, 0.5, true);
//! println!("T₃(0.5) = {}", t3);
//!
//! // Hermite polynomial (physicist's version): H₃(0.5)
//! let h3: f64 = special::hermite(3, 0.5);
//! println!("H₃(0.5) = {}", h3);
//! ```
//!
//! ## Airy Functions
//!
//! ```
//! use numrs2::special;
//!
//! // Airy function of the first kind: Ai(x)
//! let airy_ai: f64 = special::ai(1.0);
//! println!("Ai(1) = {}", airy_ai);
//!
//! // Airy function of the second kind: Bi(x)
//! let airy_bi: f64 = special::bi(1.0);
//! println!("Bi(1) = {}", airy_bi);
//! ```
//!
//! # Mathematical Background
//!
//! Special functions arise naturally in solutions to differential equations and are fundamental
//! in many areas of mathematics, physics, and engineering:
//!
//! - **Gamma/Beta**: Extend factorials to real/complex numbers, used in probability distributions
//! - **Error functions**: Probability integrals, appear in statistics and diffusion equations
//! - **Bessel functions**: Solutions to Bessel's differential equation, used in wave propagation
//! - **Elliptic integrals**: Arc length of ellipse, pendulum motion, classical mechanics
//! - **Orthogonal polynomials**: Quantum mechanics, approximation theory, numerical analysis
//! - **Airy functions**: Wave propagation, quantum mechanics, optics
//!
//! # Performance Notes
//!
//! - Functions use numerically stable algorithms (e.g., log-gamma for large arguments)
//! - Implementations leverage special algorithms (continued fractions, series expansions)
//! - For high-precision needs, consider enabling the `high-precision` feature
//! - Parallel computation available for array operations with `parallel` feature

// Re-export all scirs2-special modules
pub use scirs2_special::*;

// Additional NumRS2-specific convenience functions and aliases can be added here

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_function() {
        // Test gamma function: Γ(4) = 3! = 6
        let result: f64 = gamma(4.0);
        assert!((result - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_gamma() {
        // Test log gamma: log Γ(5) = log(4!) = log(24)
        let result = loggamma(5.0);
        let expected = 24.0_f64.ln();
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_beta_function() {
        // Test beta function: B(2,3) = Γ(2)Γ(3)/Γ(5) = 1·2/24 = 1/12
        let result: f64 = beta(2.0, 3.0);
        assert!((result - 1.0 / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_erf() {
        // Test erf(0) = 0
        let result: f64 = erf(0.0);
        assert!(result.abs() < 1e-10);

        // Test erf is odd: erf(-x) = -erf(x)
        let x = 1.5;
        let erf_x: f64 = erf(x);
        let erf_neg_x: f64 = erf(-x);
        assert!((erf_x + erf_neg_x).abs() < 1e-10);
    }

    #[test]
    fn test_erfc() {
        // Test erfc(x) + erf(x) = 1
        let x = 1.0;
        let erf_x: f64 = erf(x);
        let erfc_x: f64 = erfc(x);
        assert!((erf_x + erfc_x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bessel_j0() {
        // Test J₀(0) = 1
        let result: f64 = j0(0.0);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bessel_j1() {
        // Test J₁ exists and basic properties
        let x = 2.0;
        let result: f64 = j1(x);
        assert!(result.is_finite());
    }

    #[test]
    fn test_bessel_i0() {
        // Test I₀(0) = 1
        let result: f64 = i0(0.0);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_legendre_polynomial() {
        // Test P₀(x) = 1
        let result: f64 = legendre(0, 0.5);
        assert!((result - 1.0).abs() < 1e-10);

        // Test P₁(x) = x
        let x: f64 = 0.7;
        let result: f64 = legendre(1, x);
        assert!((result - x).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_polynomial() {
        // Test T₀(x) = 1 (first kind)
        let result: f64 = chebyshev(0, 0.5, true);
        assert!((result - 1.0).abs() < 1e-10);

        // Test T₁(x) = x (first kind)
        let x: f64 = 0.7;
        let result: f64 = chebyshev(1, x, true);
        assert!((result - x).abs() < 1e-10);
    }

    #[test]
    fn test_hermite_polynomial() {
        // Test H₀(x) = 1
        let result: f64 = hermite(0, 0.5);
        assert!((result - 1.0).abs() < 1e-10);

        // Test H₁(x) = 2x
        let x: f64 = 0.7;
        let result: f64 = hermite(1, x);
        assert!((result - 2.0 * x).abs() < 1e-10);
    }

    #[test]
    fn test_laguerre_polynomial() {
        // Test L₀(x) = 1
        let result: f64 = laguerre(0, 0.5);
        assert!((result - 1.0).abs() < 1e-10);

        // Test L₁(x) = 1 - x
        let x: f64 = 0.3;
        let result: f64 = laguerre(1, x);
        assert!((result - (1.0 - x)).abs() < 1e-10);
    }

    #[test]
    fn test_airy_ai() {
        // Test that Airy functions exist and are finite
        let result: f64 = ai(1.0);
        assert!(result.is_finite());
        assert!(result > 0.0); // Ai(x) > 0 for x > 0
    }

    #[test]
    fn test_airy_bi() {
        // Test that Airy Bi function exists and is finite
        let result: f64 = bi(1.0);
        assert!(result.is_finite());
        assert!(result > 0.0); // Bi(x) > 0 for all x
    }

    #[test]
    fn test_elliptic_k() {
        // Test complete elliptic integral K(0) = π/2
        let result: f64 = elliptic_k(0.0);
        let expected = std::f64::consts::PI / 2.0;
        assert!((result - expected).abs() < 1e-8);
    }

    #[test]
    fn test_elliptic_e() {
        // Test complete elliptic integral E(0) = π/2
        let result: f64 = elliptic_e(0.0);
        let expected = std::f64::consts::PI / 2.0;
        assert!((result - expected).abs() < 1e-8);
    }

    #[test]
    fn test_hypergeometric_1f1() {
        // Test 1F1(0, b, x) = 1
        let result: f64 = hyp1f1(0.0, 2.0, 0.5).expect("hyp1f1 should succeed");
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypergeometric_2f1() {
        // Test 2F1(a, b, c, 0) = 1
        let result: f64 = hyp2f1(1.0, 2.0, 3.0, 0.0).expect("hyp2f1 should succeed");
        assert!((result - 1.0).abs() < 1e-10);
    }
}
