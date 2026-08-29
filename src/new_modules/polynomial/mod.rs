//! Polynomial functionality and interpolation for NumRS2
//!
//! This module provides comprehensive polynomial operations including:
//! - Basic polynomial operations (evaluation, derivatives, integrals)
//! - Arithmetic operations (addition, subtraction, multiplication, division)
//! - Polynomial fitting and evaluation
//! - Root finding and construction from roots
//! - Interpolation methods (Lagrange, Newton, cubic splines)
//! - Special/orthogonal polynomials (Chebyshev, Legendre, Hermite, Laguerre, Jacobi)
//! - Utility functions (Vandermonde matrices, companion matrices, etc.)

pub mod arithmetic;
pub mod classes;
pub mod core;
pub mod fitting;
pub mod interpolation;
pub mod roots;
pub mod special;
pub mod utils;

// Re-export core types
pub use core::Polynomial;

// Re-export the NumPy `numpy.polynomial` class family (Chebyshev, Legendre,
// Hermite, HermiteE, Laguerre). The power-basis class in that same family,
// `classes::Polynomial`, is deliberately NOT re-exported under the bare name
// `Polynomial` here -- it would collide with the `core::Polynomial` type
// above (this module's pre-existing `numpy.poly1d`-style API); reach it via
// `classes::Polynomial` instead.
pub use classes::{Basis, Chebyshev, Hermite, HermiteE, Laguerre, Legendre};

// Re-export arithmetic functions
pub use arithmetic::{polyadd, polydiv, polymul, polysub};

// Re-export fitting functions
pub use fitting::{polyder, polyextrap, polyfit, polyfit_weighted, polyint, polyresidual, polyval};

// Re-export root functions
pub use roots::{poly, polyfromroots, roots};

// Re-export interpolation types and functions
pub use interpolation::{CubicSpline, PolynomialInterpolation};

// Re-export special polynomial functions
pub use special::{
    polychebyshev, polyhermite, polyjacobi, polylaguerre, polylegendre, OrthogonalPolynomials,
};

// Re-export utility functions
pub use utils::{
    polycompanion, polycompose, polygcd, polygrid2d, polymulx, polypower, polyscale, polytrim,
    polyval2d, polyvander, polyvander2d,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use approx::assert_relative_eq;

    #[test]
    fn test_polynomial_module_integration() {
        // Test that modules work together
        let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array::from_vec(vec![1.0, 6.0, 15.0, 28.0, 45.0]);

        // Fit polynomial
        let p = polyfit(&x, &y, 2).expect("Polynomial fitting should succeed");

        // Evaluate at new points
        let new_x = Array::from_vec(vec![5.0, 6.0]);
        let result = polyval(&p, &new_x).expect("Polynomial evaluation should succeed");

        // Check extrapolation
        assert_relative_eq!(result.to_vec()[0], 66.0, epsilon = 1e-8);
        assert_relative_eq!(result.to_vec()[1], 91.0, epsilon = 1e-8);
    }

    #[test]
    fn test_special_polynomials_integration() {
        // Test Chebyshev and Legendre work correctly
        let t2 =
            polychebyshev::<f64>(2, 1).expect("Chebyshev polynomial generation should succeed");
        let p2 = polylegendre::<f64>(2).expect("Legendre polynomial generation should succeed");

        // T_2(x) = 2x^2 - 1
        assert_eq!(t2.len(), 3);
        assert_relative_eq!(t2.to_vec()[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(t2.to_vec()[2], -1.0, epsilon = 1e-10);

        // P_2(0) = -1/2
        let p2_poly = Polynomial::new(p2.to_vec());
        assert_relative_eq!(p2_poly.evaluate(0.0), -0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolation_integration() {
        // Test Lagrange interpolation works with polynomial operations
        let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
        let y = Array::from_vec(vec![1.0, 4.0, 9.0]); // y = (x+1)^2

        let p = PolynomialInterpolation::lagrange(&x, &y)
            .expect("Lagrange interpolation should succeed");

        // Derivative should be 2x + 2 at x=1 -> 4
        let dp = p.derivative();
        assert_relative_eq!(dp.evaluate(1.0), 4.0, epsilon = 1e-8);
    }
}
