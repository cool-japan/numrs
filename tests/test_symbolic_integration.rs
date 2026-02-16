//! Tests for integration between symbolic and automatic differentiation

use numrs2::symbolic::*;
use std::collections::HashMap;

#[test]
fn test_symbolic_then_numeric_eval() {
    let x = Expr::var("x");
    let expr = x.clone().pow(2.0) + x.clone() * 2.0 + 1.0;

    // Symbolic differentiation
    let derivative = differentiate(&expr, "x").expect("differentiation failed");
    let simplified = simplify(&derivative);

    // Numeric evaluation
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    let numeric_result = simplified.eval(&vars).expect("evaluation failed");

    // d/dx(x² + 2x + 1) = 2x + 2, at x=3: 8
    assert_eq!(numeric_result, 8.0);
}

#[test]
fn test_compare_symbolic_numeric_derivative() {
    let x = Expr::var("x");
    // f(x) = x³ - 2x² + x - 5
    let expr = x.clone().pow(3.0) - x.clone().pow(2.0) * 2.0 + x.clone() - 5.0;

    // Symbolic derivative
    let symbolic_deriv = differentiate(&expr, "x").expect("differentiation failed");
    let simplified = simplify(&symbolic_deriv);

    // Evaluate at multiple points
    for test_x in &[1.0, 2.0, 3.0, -1.0, 0.5] {
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), *test_x);

        let symbolic_result = simplified.eval(&vars).expect("evaluation failed");

        // Numeric derivative using finite differences
        let h = 1e-8;
        let mut vars_plus = HashMap::new();
        vars_plus.insert("x".to_string(), test_x + h);
        let mut vars_minus = HashMap::new();
        vars_minus.insert("x".to_string(), test_x - h);

        let f_plus = expr.eval(&vars_plus).expect("evaluation failed");
        let f_minus = expr.eval(&vars_minus).expect("evaluation failed");
        let numeric_deriv = (f_plus - f_minus) / (2.0 * h);

        // They should agree to high precision
        assert!((symbolic_result - numeric_deriv).abs() < 1e-6);
    }
}

#[test]
fn test_gradient_evaluation() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    // f(x, y) = x² + y² + xy
    let f = x.clone().pow(2.0) + y.clone().pow(2.0) + x.clone() * y.clone();

    let grad = gradient(&f, &["x", "y"]).expect("gradient computation failed");

    let test_point = (2.0, 3.0);
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), test_point.0);
    vars.insert("y".to_string(), test_point.1);

    let dx = grad[0].eval(&vars).expect("evaluation failed");
    let dy = grad[1].eval(&vars).expect("evaluation failed");

    // ∂f/∂x = 2x + y = 4 + 3 = 7
    // ∂f/∂y = 2y + x = 6 + 2 = 8
    assert_eq!(dx, 7.0);
    assert_eq!(dy, 8.0);
}

#[test]
fn test_optimization_with_symbolic_derivatives() {
    let x = Expr::var("x");
    // f(x) = (x - 2)²
    let f = (x.clone() - 2.0).pow(2.0);

    // Find minimum by setting derivative to zero
    let df = differentiate(&f, "x").expect("differentiation failed");
    let simplified = simplify(&df);

    // df/dx = 2(x - 2) = 2x - 4
    // Setting to zero: 2x - 4 = 0 => x = 2

    // Verify derivative is zero at x = 2
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    let deriv_at_min = simplified.eval(&vars).expect("evaluation failed");
    assert!(deriv_at_min.abs() < 1e-10);
}

#[test]
fn test_chain_rule_verification() {
    let x = Expr::var("x");
    // f(x) = sin(x²)
    let f = (x.clone().pow(2.0)).sin();

    let df = differentiate(&f, "x").expect("differentiation failed");

    // df/dx = cos(x²) * 2x

    let test_x = 1.0;
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), test_x);

    let symbolic_result = df.eval(&vars).expect("evaluation failed");

    // Expected: cos(1²) * 2*1 = cos(1) * 2
    let expected = (test_x * test_x).cos() * 2.0 * test_x;

    assert!((symbolic_result - expected).abs() < 1e-10);
}

#[test]
fn test_product_rule_verification() {
    let x = Expr::var("x");
    // f(x) = x * sin(x)
    let f = x.clone() * x.clone().sin();

    let df = differentiate(&f, "x").expect("differentiation failed");

    // df/dx = sin(x) + x*cos(x)

    let test_x = 0.5;
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), test_x);

    let symbolic_result = df.eval(&vars).expect("evaluation failed");
    let expected = test_x.sin() + test_x * test_x.cos();

    assert!((symbolic_result - expected).abs() < 1e-10);
}

#[test]
fn test_quotient_rule_verification() {
    let x = Expr::var("x");
    // f(x) = x / (x + 1)
    let f = x.clone() / (x.clone() + 1.0);

    let df = differentiate(&f, "x").expect("differentiation failed");

    // df/dx = ((x+1) - x) / (x+1)² = 1 / (x+1)²

    let test_x = 2.0;
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), test_x);

    let symbolic_result = df.eval(&vars).expect("evaluation failed");
    let expected = 1.0 / ((test_x + 1.0) * (test_x + 1.0));

    assert!((symbolic_result - expected).abs() < 1e-10);
}

#[test]
fn test_simplification_after_differentiation() {
    let x = Expr::var("x");
    // f(x) = x³ + 3x² + 3x + 1 = (x+1)³
    let f = x.clone().pow(3.0) + x.clone().pow(2.0) * 3.0 + x.clone() * 3.0 + 1.0;

    let df = differentiate(&f, "x").expect("differentiation failed");
    let simplified = simplify(&df);

    // df/dx = 3x² + 6x + 3 = 3(x² + 2x + 1) = 3(x+1)²

    let test_x = 1.0;
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), test_x);

    let result = simplified.eval(&vars).expect("evaluation failed");
    // At x=1: 3(1+1)² = 3*4 = 12
    assert_eq!(result, 12.0);
}

#[test]
fn test_second_derivative_verification() {
    let x = Expr::var("x");
    // f(x) = x⁴
    let f = x.clone().pow(4.0);

    let df = differentiate(&f, "x").expect("first differentiation failed");
    let d2f = differentiate(&df, "x").expect("second differentiation failed");

    // d²f/dx² = 12x²

    let test_x = 2.0;
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), test_x);

    let result = d2f.eval(&vars).expect("evaluation failed");
    // At x=2: 12*4 = 48
    assert_eq!(result, 48.0);
}

#[test]
fn test_symbolic_matrix_differentiation() {
    let x = Expr::var("x");
    let data = vec![
        vec![x.clone().pow(2.0), x.clone()],
        vec![x.clone(), Expr::constant(1.0)],
    ];

    let mat = SymbolicMatrix::from_vec(data).expect("matrix creation failed");

    // Differentiate each element
    let mut deriv_data = vec![];
    for i in 0..mat.nrows() {
        let mut row = vec![];
        for j in 0..mat.ncols() {
            if let Some(elem) = mat.get(i, j) {
                let d_elem = differentiate(elem, "x").expect("differentiation failed");
                row.push(d_elem);
            }
        }
        deriv_data.push(row);
    }

    let deriv_mat = SymbolicMatrix::from_vec(deriv_data).expect("matrix creation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    let result = deriv_mat.eval(&vars).expect("evaluation failed");
    // d/dx [[x², x], [x, 1]] = [[2x, 1], [1, 0]], at x=3: [[6, 1], [1, 0]]
    assert_eq!(result[[0, 0]], 6.0);
    assert_eq!(result[[0, 1]], 1.0);
    assert_eq!(result[[1, 0]], 1.0);
    assert_eq!(result[[1, 1]], 0.0);
}

#[test]
fn test_taylor_series_approximation() {
    let x = Expr::var("x");
    // f(x) = exp(x) around x = 0
    let f = x.clone().exp();

    // Compute derivatives at x = 0
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);

    // Taylor series: f(x) = f(0) + f'(0)x + f''(0)x²/2 + ...
    let f0 = f.eval(&vars).expect("evaluation failed");

    let df = differentiate(&f, "x").expect("differentiation failed");
    let f1 = df.eval(&vars).expect("evaluation failed");

    let d2f = differentiate(&df, "x").expect("differentiation failed");
    let f2 = d2f.eval(&vars).expect("evaluation failed");

    // For exp(x), all derivatives at 0 are 1
    assert_eq!(f0, 1.0);
    assert_eq!(f1, 1.0);
    assert_eq!(f2, 1.0);
}

#[test]
fn test_implicit_differentiation() {
    // For circle x² + y² = r²
    // dy/dx = -x/y

    let x = Expr::var("x");
    let y = Expr::var("y");
    let r = Expr::constant(5.0);

    // x² + y² - r² = 0
    let implicit = x.clone().pow(2.0) + y.clone().pow(2.0) - r.pow(2.0);

    // d/dx(x² + y² - r²) = 2x + 2y*dy/dx = 0
    // dy/dx = -2x / (2y) = -x/y

    let dx_part = differentiate(&implicit, "x").expect("differentiation failed");
    let dy_part = differentiate(&implicit, "y").expect("differentiation failed");

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    vars.insert("y".to_string(), 4.0);

    let dx_val = dx_part.eval(&vars).expect("evaluation failed");
    let dy_val = dy_part.eval(&vars).expect("evaluation failed");

    // dy/dx = -dx_val / dy_val = -6 / 8 = -0.75
    let dydx = -dx_val / dy_val;
    assert_eq!(dydx, -0.75);
}
