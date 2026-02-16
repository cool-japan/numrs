//! Tests for symbolic differentiation

use numrs2::symbolic::*;
use std::collections::HashMap;

#[test]
fn test_diff_constant() {
    let c = Expr::constant(5.0);
    let dc = differentiate(&c, "x").expect("differentiation failed");

    assert!(matches!(dc, Expr::Constant(0.0)));
}

#[test]
fn test_diff_variable() {
    let x = Expr::var("x");
    let dx = differentiate(&x, "x").expect("differentiation failed");

    assert!(matches!(dx, Expr::Constant(1.0)));

    let y = Expr::var("y");
    let dy = differentiate(&x, "y").expect("differentiation failed");
    assert!(matches!(dy, Expr::Constant(0.0)));
}

#[test]
fn test_diff_linear() {
    let x = Expr::var("x");
    let expr = x.clone() * 3.0 + 2.0;
    let derivative = differentiate(&expr, "x").expect("differentiation failed");
    let simplified = simplify(&derivative);

    let vars = HashMap::new();
    let result = simplified.eval(&vars).expect("evaluation failed");

    // d/dx(3x + 2) = 3
    assert_eq!(result, 3.0);
}

#[test]
fn test_diff_quadratic() {
    let x = Expr::var("x");
    let expr = x.clone().pow(2.0);
    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(x²) = 2x, at x=3: 6
    assert_eq!(result, 6.0);
}

#[test]
fn test_diff_cubic() {
    let x = Expr::var("x");
    let expr = x.clone().pow(3.0);
    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(x³) = 3x², at x=2: 12
    assert_eq!(result, 12.0);
}

#[test]
fn test_diff_sum_rule() {
    let x = Expr::var("x");
    let expr = x.clone() + x.clone();
    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let vars = HashMap::new();
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(x + x) = 1 + 1 = 2
    assert_eq!(result, 2.0);
}

#[test]
fn test_diff_product_rule() {
    let x = Expr::var("x");
    let expr = x.clone() * x.clone(); // x²
    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(x * x) = x + x = 2x, at x=3: 6
    assert_eq!(result, 6.0);
}

#[test]
fn test_diff_quotient_rule() {
    let x = Expr::var("x");
    let expr = x.clone() / (x.clone() + 1.0);
    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(x / (x+1)) = 1 / (x+1)², at x=1: 1/4 = 0.25
    assert_eq!(result, 0.25);
}

#[test]
fn test_diff_sin() {
    let x = Expr::var("x");
    let expr = x.clone().sin();
    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(sin(x)) = cos(x), at x=0: 1
    assert_eq!(result, 1.0);
}

#[test]
fn test_diff_cos() {
    let x = Expr::var("x");
    let expr = x.clone().cos();
    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(cos(x)) = -sin(x), at x=0: 0
    assert_eq!(result, 0.0);
}

#[test]
fn test_diff_exp() {
    let x = Expr::var("x");
    let expr = x.clone().exp();
    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(exp(x)) = exp(x), at x=0: 1
    assert_eq!(result, 1.0);
}

#[test]
fn test_diff_ln() {
    let x = Expr::var("x");
    let expr = x.clone().ln();
    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(ln(x)) = 1/x, at x=1: 1
    assert_eq!(result, 1.0);
}

#[test]
fn test_diff_sqrt() {
    let x = Expr::var("x");
    let expr = x.clone().sqrt();
    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 4.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(sqrt(x)) = 1/(2*sqrt(x)), at x=4: 1/(2*2) = 0.25
    assert_eq!(result, 0.25);
}

#[test]
fn test_diff_chain_rule() {
    let x = Expr::var("x");
    let inner = x.clone() * 2.0;
    let expr = inner.sin(); // sin(2x)

    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(sin(2x)) = cos(2x) * 2, at x=0: 2
    assert_eq!(result, 2.0);
}

#[test]
fn test_diff_nested_chain_rule() {
    let x = Expr::var("x");
    let expr = (x.clone().pow(2.0)).sin(); // sin(x²)

    let derivative = differentiate(&expr, "x").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);
    let result = derivative.eval(&vars).expect("evaluation failed");

    // d/dx(sin(x²)) = cos(x²) * 2x, at x=0: 0
    assert_eq!(result, 0.0);
}

#[test]
fn test_gradient_2d() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let f = x.clone().pow(2.0) + y.clone().pow(2.0); // x² + y²

    let grad = gradient(&f, &["x", "y"]).expect("gradient computation failed");
    assert_eq!(grad.len(), 2);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    vars.insert("y".to_string(), 4.0);

    let dx = grad[0].eval(&vars).expect("evaluation failed");
    let dy = grad[1].eval(&vars).expect("evaluation failed");

    // ∇f = [2x, 2y], at (3, 4): [6, 8]
    assert_eq!(dx, 6.0);
    assert_eq!(dy, 8.0);
}

#[test]
fn test_gradient_multivariate() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let z = Expr::var("z");
    let f = x.clone() * y.clone() + y.clone() * z.clone() + z.clone() * x.clone();

    let grad = gradient(&f, &["x", "y", "z"]).expect("gradient computation failed");
    assert_eq!(grad.len(), 3);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);
    vars.insert("y".to_string(), 2.0);
    vars.insert("z".to_string(), 3.0);

    let dx = grad[0].eval(&vars).expect("evaluation failed");
    let dy = grad[1].eval(&vars).expect("evaluation failed");
    let dz = grad[2].eval(&vars).expect("evaluation failed");

    // ∂f/∂x = y + z = 5
    // ∂f/∂y = x + z = 4
    // ∂f/∂z = y + x = 3
    assert_eq!(dx, 5.0);
    assert_eq!(dy, 4.0);
    assert_eq!(dz, 3.0);
}

#[test]
fn test_jacobian() {
    let x = Expr::var("x");
    let y = Expr::var("y");

    let f1 = x.clone().pow(2.0); // x²
    let f2 = y.clone().pow(2.0); // y²

    let exprs = vec![f1, f2];
    let jac = jacobian(&exprs, &["x", "y"]).expect("jacobian computation failed");

    assert_eq!(jac.len(), 2);
    assert_eq!(jac[0].len(), 2);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);
    vars.insert("y".to_string(), 3.0);

    // J = [[2x, 0], [0, 2y]]
    let j00 = jac[0][0].eval(&vars).expect("eval failed");
    let j01 = jac[0][1].eval(&vars).expect("eval failed");
    let j10 = jac[1][0].eval(&vars).expect("eval failed");
    let j11 = jac[1][1].eval(&vars).expect("eval failed");

    assert_eq!(j00, 4.0); // 2*2
    assert_eq!(j01, 0.0);
    assert_eq!(j10, 0.0);
    assert_eq!(j11, 6.0); // 2*3
}

#[test]
fn test_hessian() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let f = x.clone().pow(2.0) + x.clone() * y.clone() + y.clone().pow(2.0);

    let hess = hessian(&f, &["x", "y"]).expect("hessian computation failed");

    assert_eq!(hess.len(), 2);
    assert_eq!(hess[0].len(), 2);

    // Simplify each element of the Hessian
    let h00_simp = simplify(&hess[0][0]);
    let h01_simp = simplify(&hess[0][1]);
    let h10_simp = simplify(&hess[1][0]);
    let h11_simp = simplify(&hess[1][1]);

    let vars = HashMap::new();

    // H = [[2, 1], [1, 2]]
    let h00 = h00_simp.eval(&vars).expect("eval failed");
    let h01 = h01_simp.eval(&vars).expect("eval failed");
    let h10 = h10_simp.eval(&vars).expect("eval failed");
    let h11 = h11_simp.eval(&vars).expect("eval failed");

    assert_eq!(h00, 2.0);
    assert_eq!(h01, 1.0);
    assert_eq!(h10, 1.0);
    assert_eq!(h11, 2.0);
}

#[test]
fn test_directional_derivative() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let f = x.clone().pow(2.0) + y.clone().pow(2.0);

    let direction = vec![Expr::constant(1.0), Expr::constant(0.0)];
    let dir_deriv = directional_derivative(&f, &["x", "y"], &direction)
        .expect("directional derivative computation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    vars.insert("y".to_string(), 4.0);

    let result = dir_deriv.eval(&vars).expect("evaluation failed");

    // ∇f · [1, 0] = [2x, 2y] · [1, 0] = 2x = 6
    assert_eq!(result, 6.0);
}

#[test]
fn test_second_derivative() {
    let x = Expr::var("x");
    let expr = x.clone().pow(3.0); // x³

    let first_deriv = differentiate(&expr, "x").expect("first differentiation failed");
    let second_deriv = differentiate(&first_deriv, "x").expect("second differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    let result = second_deriv.eval(&vars).expect("evaluation failed");

    // d²/dx²(x³) = 6x, at x=2: 12
    assert_eq!(result, 12.0);
}

#[test]
fn test_diff_power_general() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let expr = x.clone().pow(y.clone()); // x^y

    let dx = differentiate(&expr, "x").expect("differentiation failed");
    let dy = differentiate(&expr, "y").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);
    vars.insert("y".to_string(), 3.0);

    let dx_val = dx.eval(&vars).expect("evaluation failed");
    let dy_val = dy.eval(&vars).expect("evaluation failed");

    // ∂/∂x(x^y) = y * x^(y-1) = 3 * 2² = 12
    assert_eq!(dx_val, 12.0);

    // ∂/∂y(x^y) = x^y * ln(x)
    let expected_dy = 8.0_f64 * 2.0_f64.ln();
    assert!((dy_val - expected_dy).abs() < 1e-10);
}
