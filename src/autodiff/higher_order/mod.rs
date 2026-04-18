//! Higher-order automatic differentiation for NumRS2
//!
//! This module provides exact higher-order derivatives through hyper-dual numbers,
//! efficient Jacobian and Hessian computation, and gradient validation utilities.
//!
//! # Overview
//!
//! ## HyperDual Numbers
//!
//! Hyper-dual numbers extend dual numbers to capture second-order derivatives exactly.
//! A hyper-dual number has four components: `f + f_1*e1 + f_2*e2 + f_12*e1e2`
//! where `e1^2 = e2^2 = 0` but `e1*e2 != 0`. The `e1*e2` component gives the
//! exact mixed second partial derivative `d^2f/(dx_i dx_j)`.
//!
//! ## Jacobian Computation
//!
//! For vector-valued functions `f: R^n -> R^m`:
//! - **Forward mode**: Efficient when `n < m` (few inputs, many outputs)
//! - **Reverse mode**: Efficient when `n > m` (many inputs, few outputs)
//! - **Auto mode**: Automatically selects the most efficient mode
//!
//! ## Hessian Computation
//!
//! For scalar functions `f: R^n -> R`:
//! - **Exact (HyperDual)**: Analytical second derivatives, no numerical error
//! - **Forward-over-reverse**: Exact first derivatives, numerical second derivatives
//! - **Hessian-vector product**: Efficient `Hv` without forming full matrix
//!
//! ## Gradient Checking
//!
//! Validates analytical gradients against numerical finite differences.
//!
//! # Examples
//!
//! ```rust,ignore
//! use numrs2::autodiff::higher_order::*;
//!
//! // Compute exact Hessian of f(x,y) = x^2 + x*y + y^2
//! fn f(vars: &[HyperDual<f64>]) -> HyperDual<f64> {
//!     let x = vars[0];
//!     let y = vars[1];
//!     x * x + x * y + y * y
//! }
//!
//! let point = vec![1.0, 2.0];
//! let hess = hessian_exact(f, &point).expect("valid hessian computation");
//! // H = [[2, 1], [1, 2]]
//! ```

pub mod gradient_utils;
pub mod hessian;
pub mod hyperdual;
pub mod jacobian;
pub mod taylor;

pub use gradient_utils::{
    compute_gradient_ad, compute_gradient_hyperdual, gradient_check, gradient_check_ad,
    GradientCheckResult,
};
pub use hessian::{hessian_exact, hessian_forward_over_reverse, hessian_vector_product};
pub use hyperdual::HyperDual;
pub use jacobian::{forward_jacobian, jacobian_auto, reverse_jacobian};
pub use taylor::{multivariate_taylor, TaylorExpansion2};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::{Dual, Tape, Var};

    const TOL: f64 = 1e-10;
    const NUMERICAL_TOL: f64 = 1e-5;

    // ====================================================================
    // HyperDual Number Tests
    // ====================================================================

    #[test]
    fn test_hyperdual_basic_arithmetic() {
        // Test addition: (a + e1) + (b + e2) = (a+b) + e1 + e2
        let x = HyperDual::<f64>::new(3.0, 1.0, 0.0, 0.0);
        let y = HyperDual::<f64>::new(4.0, 0.0, 1.0, 0.0);

        let sum = x + y;
        assert!((sum.real() - 7.0).abs() < TOL);
        assert!((sum.eps1() - 1.0).abs() < TOL);
        assert!((sum.eps2() - 1.0).abs() < TOL);
        assert!((sum.eps1eps2() - 0.0).abs() < TOL);

        // Test subtraction
        let diff = x - y;
        assert!((diff.real() - (-1.0)).abs() < TOL);

        // Test multiplication: x*y where x = 3+e1, y = 4+e2
        // real: 12, e1: 4, e2: 3, e1e2: 1 (product rule cross term)
        let prod = x * y;
        assert!((prod.real() - 12.0).abs() < TOL);
        assert!((prod.eps1() - 4.0).abs() < TOL);
        assert!((prod.eps2() - 3.0).abs() < TOL);
        assert!((prod.eps1eps2() - 1.0).abs() < TOL);

        // Test negation
        let neg = -x;
        assert!((neg.real() - (-3.0)).abs() < TOL);
        assert!((neg.eps1() - (-1.0)).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_division() {
        // f(x,y) = x/y at (6, 3)
        // d^2f/dxdy = -1/y^2 = -1/9
        let x = HyperDual::<f64>::new(6.0, 1.0, 0.0, 0.0);
        let y = HyperDual::<f64>::new(3.0, 0.0, 1.0, 0.0);

        let quot = x / y;
        assert!((quot.real() - 2.0).abs() < TOL);
        // df/dx = 1/y = 1/3
        assert!((quot.eps1() - 1.0 / 3.0).abs() < TOL);
        // df/dy = -x/y^2 = -6/9 = -2/3
        assert!((quot.eps2() - (-2.0 / 3.0)).abs() < TOL);
        // d^2f/dxdy = -1/y^2 = -1/9
        assert!((quot.eps1eps2() - (-1.0 / 9.0)).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_exp() {
        // f(x) = e^x, f'(x) = e^x, f''(x) = e^x
        let x = HyperDual::<f64>::new(2.0, 1.0, 1.0, 0.0);
        let result = x.exp();

        let e2 = 2.0_f64.exp();
        assert!((result.real() - e2).abs() < TOL);
        assert!((result.eps1() - e2).abs() < TOL);
        assert!((result.eps2() - e2).abs() < TOL);
        // f''(x)*b*c + f'(x)*d = e^2 * 1*1 + e^2 * 0 = e^2
        assert!((result.eps1eps2() - e2).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_sin_cos() {
        let val = 1.0_f64;
        let x = HyperDual::new(val, 1.0, 1.0, 0.0);

        // sin(x): f' = cos(x), f'' = -sin(x)
        let sin_result = x.sin();
        assert!((sin_result.real() - val.sin()).abs() < TOL);
        assert!((sin_result.eps1() - val.cos()).abs() < TOL);
        assert!((sin_result.eps1eps2() - (-val.sin())).abs() < TOL);

        // cos(x): f' = -sin(x), f'' = -cos(x)
        let cos_result = x.cos();
        assert!((cos_result.real() - val.cos()).abs() < TOL);
        assert!((cos_result.eps1() - (-val.sin())).abs() < TOL);
        assert!((cos_result.eps1eps2() - (-val.cos())).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_chain_rule() {
        // f(x) = sin(x^2) at x = 1.5
        // f'(x) = cos(x^2) * 2x
        // f''(x) = -sin(x^2) * (2x)^2 + cos(x^2) * 2
        //        = -4x^2*sin(x^2) + 2*cos(x^2)
        let val = 1.5_f64;
        let x = HyperDual::new(val, 1.0, 1.0, 0.0);

        let x_sq = x * x;
        let result = x_sq.sin();

        let x2 = val * val;
        let expected_val = x2.sin();
        let expected_deriv = x2.cos() * 2.0 * val;
        let expected_second = -4.0 * x2 * x2.sin() + 2.0 * x2.cos();

        assert!((result.real() - expected_val).abs() < TOL);
        assert!((result.eps1() - expected_deriv).abs() < TOL);
        assert!((result.eps1eps2() - expected_second).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_powi() {
        // f(x) = x^3 at x = -2
        // f(-2) = -8, f'(-2) = 12, f''(-2) = -12
        let x = HyperDual::<f64>::new(-2.0, 1.0, 1.0, 0.0);
        let result = x.powi(3);

        assert!((result.real() - (-8.0)).abs() < TOL);
        assert!((result.eps1() - 12.0).abs() < TOL);
        // f''(x) = 6x, at x=-2: -12
        assert!((result.eps1eps2() - (-12.0)).abs() < TOL);
    }

    // ====================================================================
    // Forward Jacobian Tests
    // ====================================================================

    #[test]
    fn test_forward_jacobian_linear() {
        // f(x,y) = [x + 2y, 3x + 4y]
        // J = [[1, 2], [3, 4]]
        let f = |vars: &[Dual<f64>]| -> Vec<Dual<f64>> {
            let two = Dual::constant(2.0);
            let three = Dual::constant(3.0);
            let four = Dual::constant(4.0);
            vec![vars[0] + two * vars[1], three * vars[0] + four * vars[1]]
        };

        let jac = forward_jacobian(f, &[1.0, 1.0]).expect("forward_jacobian should succeed");
        let jac_vec = jac.to_vec();

        // J = [[1, 2], [3, 4]] stored row-major
        assert!((jac_vec[0] - 1.0).abs() < TOL); // J[0,0]
        assert!((jac_vec[1] - 2.0).abs() < TOL); // J[0,1]
        assert!((jac_vec[2] - 3.0).abs() < TOL); // J[1,0]
        assert!((jac_vec[3] - 4.0).abs() < TOL); // J[1,1]
    }

    #[test]
    fn test_forward_jacobian_quadratic() {
        // f(x,y) = [x^2, x*y]
        // J = [[2x, 0], [y, x]]
        // At (2, 3): J = [[4, 0], [3, 2]]
        let f =
            |vars: &[Dual<f64>]| -> Vec<Dual<f64>> { vec![vars[0] * vars[0], vars[0] * vars[1]] };

        let jac = forward_jacobian(f, &[2.0, 3.0]).expect("forward_jacobian should succeed");
        let jac_vec = jac.to_vec();

        assert!((jac_vec[0] - 4.0).abs() < TOL); // 2x = 4
        assert!((jac_vec[1] - 0.0).abs() < TOL); // 0
        assert!((jac_vec[2] - 3.0).abs() < TOL); // y = 3
        assert!((jac_vec[3] - 2.0).abs() < TOL); // x = 2
    }

    // ====================================================================
    // Reverse Jacobian Tests
    // ====================================================================

    #[test]
    fn test_reverse_jacobian_linear() {
        // f(x,y) = [x + 2y, 3x + 4y]
        // J = [[1, 2], [3, 4]]
        let jac = reverse_jacobian(
            |tape: &mut Tape<f64>, vars: &[Var]| {
                let two = tape.var(2.0);
                let three = tape.var(3.0);
                let four = tape.var(4.0);

                let two_y = tape.mul(two, vars[1]);
                let out1 = tape.add(vars[0], two_y);

                let three_x = tape.mul(three, vars[0]);
                let four_y = tape.mul(four, vars[1]);
                let out2 = tape.add(three_x, four_y);

                vec![out1, out2]
            },
            &[1.0, 1.0],
        )
        .expect("reverse_jacobian should succeed");

        let jac_vec = jac.to_vec();
        assert!((jac_vec[0] - 1.0).abs() < TOL);
        assert!((jac_vec[1] - 2.0).abs() < TOL);
        assert!((jac_vec[2] - 3.0).abs() < TOL);
        assert!((jac_vec[3] - 4.0).abs() < TOL);
    }

    #[test]
    fn test_reverse_jacobian_multi_output() {
        // f(x) = [x^2, x^3] (single input, two outputs)
        // J = [[2x], [3x^2]]
        // At x=2: J = [[4], [12]]
        let jac = reverse_jacobian(
            |tape: &mut Tape<f64>, vars: &[Var]| {
                let x_sq = tape.mul(vars[0], vars[0]);
                let x_cube = tape.mul(x_sq, vars[0]);
                vec![x_sq, x_cube]
            },
            &[2.0],
        )
        .expect("reverse_jacobian should succeed");

        let jac_vec = jac.to_vec();
        assert!((jac_vec[0] - 4.0).abs() < TOL); // 2x = 4
        assert!((jac_vec[1] - 12.0).abs() < TOL); // 3x^2 = 12
    }

    // ====================================================================
    // Jacobian Agreement Test
    // ====================================================================

    #[test]
    fn test_jacobian_forward_reverse_agree() {
        // Both modes should give the same Jacobian for the same function
        // f(x,y) = [x*y + x, y^2 - x]
        let fwd_jac = forward_jacobian(
            |vars: &[Dual<f64>]| -> Vec<Dual<f64>> {
                let _one = Dual::constant(1.0);
                vec![vars[0] * vars[1] + vars[0], vars[1] * vars[1] - vars[0]]
            },
            &[2.0, 3.0],
        )
        .expect("forward_jacobian should succeed");

        let rev_jac = reverse_jacobian(
            |tape: &mut Tape<f64>, vars: &[Var]| {
                let xy = tape.mul(vars[0], vars[1]);
                let out1 = tape.add(xy, vars[0]);

                let y_sq = tape.mul(vars[1], vars[1]);
                let out2 = tape.sub(y_sq, vars[0]);

                vec![out1, out2]
            },
            &[2.0, 3.0],
        )
        .expect("reverse_jacobian should succeed");

        let fwd_vec = fwd_jac.to_vec();
        let rev_vec = rev_jac.to_vec();

        for k in 0..fwd_vec.len() {
            assert!(
                (fwd_vec[k] - rev_vec[k]).abs() < TOL,
                "Jacobian mismatch at index {}: forward={}, reverse={}",
                k,
                fwd_vec[k],
                rev_vec[k]
            );
        }
    }

    // ====================================================================
    // Hessian Tests
    // ====================================================================

    #[test]
    fn test_hessian_exact_quadratic() {
        // f(x,y) = x^2 + 3*x*y + 2*y^2
        // H = [[2, 3], [3, 4]] (constant for quadratic functions)
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            let three = HyperDual::constant(3.0);
            let two = HyperDual::constant(2.0);
            vars[0] * vars[0] + three * vars[0] * vars[1] + two * vars[1] * vars[1]
        };

        let hess = hessian_exact(f, &[5.0, 7.0]).expect("hessian_exact should succeed");
        let hess_vec = hess.to_vec();

        assert!((hess_vec[0] - 2.0).abs() < TOL); // H[0,0] = 2
        assert!((hess_vec[1] - 3.0).abs() < TOL); // H[0,1] = 3
        assert!((hess_vec[2] - 3.0).abs() < TOL); // H[1,0] = 3 (symmetric)
        assert!((hess_vec[3] - 4.0).abs() < TOL); // H[1,1] = 4
    }

    #[test]
    fn test_hessian_exact_rosenbrock() {
        // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // At (1,1):
        // H[0,0] = 2 + 1200*x^2 - 400*y = 2 + 1200 - 400 = 802
        // H[0,1] = H[1,0] = -400*x = -400
        // H[1,1] = 200
        let rosenbrock = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            let x = vars[0];
            let y = vars[1];
            let one = HyperDual::constant(1.0);
            let hundred = HyperDual::constant(100.0);
            let diff1 = one - x;
            let diff2 = y - x * x;
            diff1 * diff1 + hundred * diff2 * diff2
        };

        let hess = hessian_exact(rosenbrock, &[1.0, 1.0]).expect("hessian_exact should succeed");
        let hess_vec = hess.to_vec();

        assert!((hess_vec[0] - 802.0).abs() < TOL);
        assert!((hess_vec[1] - (-400.0)).abs() < TOL);
        assert!((hess_vec[2] - (-400.0)).abs() < TOL);
        assert!((hess_vec[3] - 200.0).abs() < TOL);
    }

    #[test]
    fn test_hessian_vector_product_simple() {
        // f(x,y) = x^2 + 2*x*y + 3*y^2
        // H = [[2, 2], [2, 6]]
        // v = [1, 2]
        // H*v = [2*1+2*2, 2*1+6*2] = [6, 14]
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            let two = HyperDual::constant(2.0);
            let three = HyperDual::constant(3.0);
            vars[0] * vars[0] + two * vars[0] * vars[1] + three * vars[1] * vars[1]
        };

        let hv = hessian_vector_product(f, &[1.0, 1.0], &[1.0, 2.0])
            .expect("hessian_vector_product should succeed");

        assert!((hv[0] - 6.0).abs() < TOL);
        assert!((hv[1] - 14.0).abs() < TOL);
    }

    #[test]
    fn test_hessian_forward_over_reverse() {
        // f(x,y) = x^2 + y^2
        // H = [[2, 0], [0, 2]]
        let f = |tape: &mut Tape<f64>, vars: &[Var]| -> Var {
            let x_sq = tape.mul(vars[0], vars[0]);
            let y_sq = tape.mul(vars[1], vars[1]);
            tape.add(x_sq, y_sq)
        };

        let hess = hessian_forward_over_reverse(f, &[3.0, 4.0])
            .expect("hessian_forward_over_reverse should succeed");
        let hess_vec = hess.to_vec();

        assert!((hess_vec[0] - 2.0).abs() < NUMERICAL_TOL); // H[0,0]
        assert!(hess_vec[1].abs() < NUMERICAL_TOL); // H[0,1]
        assert!(hess_vec[2].abs() < NUMERICAL_TOL); // H[1,0]
        assert!((hess_vec[3] - 2.0).abs() < NUMERICAL_TOL); // H[1,1]
    }

    // ====================================================================
    // Gradient Check Tests
    // ====================================================================

    #[test]
    fn test_gradient_check_passes() {
        // f(x,y) = x^2 + y^2
        // Correct gradient at (3, 4) is [6, 8]
        let f_numerical = |vars: &[f64]| -> f64 { vars[0] * vars[0] + vars[1] * vars[1] };
        let analytical = vec![6.0, 8.0];

        let result = gradient_check(f_numerical, &[3.0, 4.0], &analytical, 1e-4)
            .expect("gradient_check should succeed");

        assert!(result.passed, "Correct gradient should pass check");
        assert!(result.max_rel_error < 1e-4);
    }

    #[test]
    fn test_gradient_check_detects_error() {
        // f(x,y) = x^2 + y^2
        // Wrong gradient: [6, 100] (y-component is wrong)
        let f_numerical = |vars: &[f64]| -> f64 { vars[0] * vars[0] + vars[1] * vars[1] };
        let wrong_analytical = vec![6.0, 100.0];

        let result = gradient_check(f_numerical, &[3.0, 4.0], &wrong_analytical, 1e-4)
            .expect("gradient_check should succeed");

        assert!(!result.passed, "Wrong gradient should fail check");
        assert!(result.max_abs_error > 90.0);
    }

    #[test]
    fn test_gradient_check_ad_integration() {
        // Test the combined AD + numerical check
        let f_ad = |vars: &[Dual<f64>]| -> Dual<f64> { vars[0] * vars[0] + vars[1] * vars[1] };
        let f_num = |vars: &[f64]| -> f64 { vars[0] * vars[0] + vars[1] * vars[1] };

        let result = gradient_check_ad(f_ad, f_num, &[3.0, 4.0], 1e-4)
            .expect("gradient_check_ad should succeed");

        assert!(result.passed);
    }

    // ====================================================================
    // Taylor Series Tests
    // ====================================================================

    #[test]
    fn test_taylor_quadratic_approximation() {
        // f(x,y) = x^2 + y^2 is exactly quadratic, so the 2nd order
        // Taylor expansion should be exact everywhere
        let f =
            |vars: &[HyperDual<f64>]| -> HyperDual<f64> { vars[0] * vars[0] + vars[1] * vars[1] };

        let taylor =
            multivariate_taylor(f, &[1.0, 1.0]).expect("multivariate_taylor should succeed");

        // Value at center: f(1,1) = 2
        assert!((taylor.value - 2.0).abs() < TOL);

        // Gradient at center: [2, 2]
        assert!((taylor.gradient[0] - 2.0).abs() < TOL);
        assert!((taylor.gradient[1] - 2.0).abs() < TOL);

        // For quadratic function, Taylor expansion is exact
        let approx = taylor
            .evaluate(&[2.0, 3.0])
            .expect("evaluate should succeed");
        let exact = 4.0 + 9.0; // f(2,3) = 13
        assert!((approx - exact).abs() < TOL);
    }

    #[test]
    fn test_taylor_approximation_accuracy() {
        // f(x,y) = exp(x) * sin(y) - Taylor expansion around (0, 0)
        // should be accurate near the center
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> { vars[0].exp() * vars[1].sin() };

        let taylor =
            multivariate_taylor(f, &[0.0, 0.0]).expect("multivariate_taylor should succeed");

        // f(0,0) = exp(0)*sin(0) = 0
        assert!((taylor.value - 0.0).abs() < TOL);

        // Test at a nearby point (0.1, 0.1)
        let approx = taylor
            .evaluate(&[0.1, 0.1])
            .expect("evaluate should succeed");
        let exact = 0.1_f64.exp() * 0.1_f64.sin();

        // Second-order Taylor should be accurate to within O(h^3)
        let error = (approx - exact).abs();
        assert!(
            error < 1e-3,
            "Taylor approximation error {} should be small near center",
            error
        );
    }

    // ====================================================================
    // Edge Case Tests
    // ====================================================================

    #[test]
    fn test_hessian_1d_function() {
        // f(x) = x^3, f''(x) = 6x
        // At x=2: H = [[12]]
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> { vars[0].powi(3) };

        let hess = hessian_exact(f, &[2.0]).expect("hessian_exact should succeed for 1D");
        let hess_vec = hess.to_vec();

        assert!((hess_vec[0] - 12.0).abs() < TOL);
    }

    #[test]
    fn test_hessian_identity_function() {
        // f(x,y) = x + y (linear function)
        // H = [[0, 0], [0, 0]]
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> { vars[0] + vars[1] };

        let hess = hessian_exact(f, &[3.0, 4.0]).expect("hessian_exact should succeed");
        let hess_vec = hess.to_vec();

        for val in &hess_vec {
            assert!(val.abs() < TOL, "Hessian of linear function should be zero");
        }
    }

    #[test]
    fn test_hyperdual_ln() {
        // f(x) = ln(x) at x = 2
        // f' = 1/x = 0.5, f'' = -1/x^2 = -0.25
        let x = HyperDual::<f64>::new(2.0, 1.0, 1.0, 0.0);
        let result = x.ln();

        assert!((result.real() - 2.0_f64.ln()).abs() < TOL);
        assert!((result.eps1() - 0.5_f64).abs() < TOL);
        assert!((result.eps1eps2() - (-0.25_f64)).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_tanh() {
        // f(x) = tanh(x) at x = 0
        // f(0) = 0, f'(0) = sech^2(0) = 1, f''(0) = -2*tanh(0)*sech^2(0) = 0
        let x = HyperDual::<f64>::new(0.0, 1.0, 1.0, 0.0);
        let result = x.tanh();

        assert!((result.real() - 0.0_f64).abs() < TOL);
        assert!((result.eps1() - 1.0_f64).abs() < TOL);
        assert!((result.eps1eps2() - 0.0_f64).abs() < TOL);
    }

    #[test]
    fn test_hyperdual_sigmoid() {
        // f(x) = sigmoid(x) at x = 0
        // f(0) = 0.5, f'(0) = 0.25, f''(0) = 0
        let x = HyperDual::<f64>::new(0.0, 1.0, 1.0, 0.0);
        let result = x.sigmoid();

        assert!((result.real() - 0.5_f64).abs() < TOL);
        assert!((result.eps1() - 0.25_f64).abs() < TOL);
        // f''(0) = f'(0) * (1 - 2*f(0)) = 0.25 * (1 - 1) = 0
        assert!((result.eps1eps2() - 0.0_f64).abs() < TOL);
    }

    #[test]
    fn test_hessian_3d_function() {
        // f(x,y,z) = x*y + y*z + x*z
        // H = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            vars[0] * vars[1] + vars[1] * vars[2] + vars[0] * vars[2]
        };

        let hess = hessian_exact(f, &[1.0, 2.0, 3.0]).expect("hessian_exact should succeed for 3D");
        let hess_vec = hess.to_vec();

        // H[0,0] = 0, H[0,1] = 1, H[0,2] = 1
        assert!(hess_vec[0].abs() < TOL);
        assert!((hess_vec[1] - 1.0).abs() < TOL);
        assert!((hess_vec[2] - 1.0).abs() < TOL);
        // H[1,0] = 1, H[1,1] = 0, H[1,2] = 1
        assert!((hess_vec[3] - 1.0).abs() < TOL);
        assert!(hess_vec[4].abs() < TOL);
        assert!((hess_vec[5] - 1.0).abs() < TOL);
        // H[2,0] = 1, H[2,1] = 1, H[2,2] = 0
        assert!((hess_vec[6] - 1.0).abs() < TOL);
        assert!((hess_vec[7] - 1.0).abs() < TOL);
        assert!(hess_vec[8].abs() < TOL);
    }

    #[test]
    fn test_compute_gradient_ad_matches_hyperdual() {
        // Verify that gradient from Dual and HyperDual agree
        let f_dual = |vars: &[Dual<f64>]| -> Dual<f64> {
            vars[0] * vars[0] + vars[0] * vars[1] + vars[1] * vars[1]
        };
        let f_hyper = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            vars[0] * vars[0] + vars[0] * vars[1] + vars[1] * vars[1]
        };

        let point = &[2.0, 3.0];
        let grad_dual =
            compute_gradient_ad(f_dual, point).expect("compute_gradient_ad should succeed");
        let grad_hyper = compute_gradient_hyperdual(&f_hyper, point)
            .expect("compute_gradient_hyperdual should succeed");

        for i in 0..grad_dual.len() {
            assert!(
                (grad_dual[i] - grad_hyper[i]).abs() < TOL,
                "Gradient component {} disagrees: dual={}, hyper={}",
                i,
                grad_dual[i],
                grad_hyper[i]
            );
        }
    }

    #[test]
    fn test_hessian_vector_product_matches_full_hessian() {
        // Verify that H*v from hessian_vector_product matches H_full * v
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            let three = HyperDual::constant(3.0);
            vars[0] * vars[0] + three * vars[0] * vars[1] + vars[1] * vars[1]
        };

        let x = &[2.0, 3.0];
        let v = &[1.0, -1.0];

        // Full Hessian: H = [[2, 3], [3, 2]]
        let hess = hessian_exact(f, x).expect("hessian_exact should succeed");
        let hess_vec = hess.to_vec();

        // H*v via full matrix
        let n = x.len();
        let mut hv_full = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                hv_full[i] += hess_vec[i * n + j] * v[j];
            }
        }

        // H*v via hessian_vector_product
        let hv_efficient =
            hessian_vector_product(f, x, v).expect("hessian_vector_product should succeed");

        for i in 0..n {
            assert!(
                (hv_full[i] - hv_efficient[i]).abs() < TOL,
                "Hv component {} disagrees: full={}, efficient={}",
                i,
                hv_full[i],
                hv_efficient[i]
            );
        }
    }

    #[test]
    fn test_hessian_exact_vs_forward_over_reverse() {
        // Both methods should give approximately the same Hessian
        let f_hyper = |vars: &[HyperDual<f64>]| -> HyperDual<f64> {
            vars[0] * vars[0] + vars[0] * vars[1] + vars[1] * vars[1]
        };
        let f_tape = |tape: &mut Tape<f64>, vars: &[Var]| -> Var {
            let x_sq = tape.mul(vars[0], vars[0]);
            let xy = tape.mul(vars[0], vars[1]);
            let y_sq = tape.mul(vars[1], vars[1]);
            let tmp = tape.add(x_sq, xy);
            tape.add(tmp, y_sq)
        };

        let x = &[2.0, 3.0];
        let hess_exact = hessian_exact(f_hyper, x).expect("hessian_exact should succeed");
        let hess_for = hessian_forward_over_reverse(f_tape, x)
            .expect("hessian_forward_over_reverse should succeed");

        let exact_vec = hess_exact.to_vec();
        let for_vec = hess_for.to_vec();

        for k in 0..exact_vec.len() {
            assert!(
                (exact_vec[k] - for_vec[k]).abs() < NUMERICAL_TOL,
                "Hessian element {} disagrees: exact={}, forward_over_reverse={}",
                k,
                exact_vec[k],
                for_vec[k]
            );
        }
    }

    #[test]
    fn test_empty_input_errors() {
        // All functions should return errors for empty input
        let f_dual = |_vars: &[Dual<f64>]| -> Vec<Dual<f64>> { vec![] };
        assert!(forward_jacobian(f_dual, &[]).is_err());

        let f_hyper = |_vars: &[HyperDual<f64>]| -> HyperDual<f64> { HyperDual::constant(0.0) };
        assert!(hessian_exact(f_hyper, &[]).is_err());
        assert!(hessian_vector_product(f_hyper, &[], &[]).is_err());
        assert!(multivariate_taylor(f_hyper, &[]).is_err());
    }

    #[test]
    fn test_hessian_vector_product_dimension_mismatch() {
        let f = |vars: &[HyperDual<f64>]| -> HyperDual<f64> { vars[0] * vars[0] };

        // v has wrong dimension
        let result = hessian_vector_product(f, &[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }
}
