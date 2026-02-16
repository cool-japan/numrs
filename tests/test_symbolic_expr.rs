//! Tests for symbolic expression creation and evaluation

use numrs2::symbolic::*;
use std::collections::HashMap;

#[test]
fn test_constant_creation() {
    let c = Expr::constant(2.5);
    assert!(matches!(c, Expr::Constant(_)));
}

#[test]
fn test_variable_creation() {
    let x = Expr::var("x");
    assert!(matches!(x, Expr::Variable(_)));
}

#[test]
fn test_basic_arithmetic() {
    let x = Expr::var("x");
    let y = Expr::var("y");

    let sum = x.clone() + y.clone();
    assert!(matches!(sum, Expr::Add(_, _)));

    let diff = x.clone() - y.clone();
    assert!(matches!(diff, Expr::Sub(_, _)));

    let prod = x.clone() * y.clone();
    assert!(matches!(prod, Expr::Mul(_, _)));

    let quot = x.clone() / y.clone();
    assert!(matches!(quot, Expr::Div(_, _)));
}

#[test]
fn test_mixed_arithmetic_with_f64() {
    let x = Expr::var("x");

    let expr1 = x.clone() + 5.0;
    assert!(matches!(expr1, Expr::Add(_, _)));

    let expr2 = 3.0 * x.clone();
    assert!(matches!(expr2, Expr::Mul(_, _)));

    let expr3 = x.clone() - 2.0;
    assert!(matches!(expr3, Expr::Sub(_, _)));

    let expr4 = x.clone() / 4.0;
    assert!(matches!(expr4, Expr::Div(_, _)));
}

#[test]
fn test_power_function() {
    let x = Expr::var("x");
    let squared = x.pow(2.0);
    assert!(matches!(squared, Expr::Pow(_, _)));
}

#[test]
fn test_trigonometric_functions() {
    let x = Expr::var("x");

    let sin_x = x.clone().sin();
    assert!(matches!(sin_x, Expr::Sin(_)));

    let cos_x = x.clone().cos();
    assert!(matches!(cos_x, Expr::Cos(_)));

    let tan_x = x.clone().tan();
    assert!(matches!(tan_x, Expr::Tan(_)));
}

#[test]
fn test_exponential_and_log() {
    let x = Expr::var("x");

    let exp_x = x.clone().exp();
    assert!(matches!(exp_x, Expr::Exp(_)));

    let ln_x = x.clone().ln();
    assert!(matches!(ln_x, Expr::Ln(_)));

    let sqrt_x = x.clone().sqrt();
    assert!(matches!(sqrt_x, Expr::Sqrt(_)));
}

#[test]
fn test_negation() {
    let x = Expr::var("x");
    let neg_x = -x;
    assert!(matches!(neg_x, Expr::Neg(_)));
}

#[test]
fn test_simple_evaluation() {
    let x = Expr::var("x");
    let expr = x.clone() * 2.0 + 3.0;

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);

    let result = expr.eval(&vars).expect("evaluation failed");
    assert_eq!(result, 13.0); // 5 * 2 + 3 = 13
}

#[test]
fn test_polynomial_evaluation() {
    let x = Expr::var("x");
    // f(x) = x² + 2x + 1
    let expr = x.clone().pow(2.0) + x.clone() * 2.0 + 1.0;

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    let result = expr.eval(&vars).expect("evaluation failed");
    assert_eq!(result, 16.0); // 9 + 6 + 1 = 16
}

#[test]
fn test_multivariate_evaluation() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let expr = x.clone() * y.clone() + x.clone() + y.clone();

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);
    vars.insert("y".to_string(), 3.0);

    let result = expr.eval(&vars).expect("evaluation failed");
    assert_eq!(result, 11.0); // 2*3 + 2 + 3 = 11
}

#[test]
fn test_trigonometric_evaluation() {
    let x = Expr::var("x");
    let expr = x.clone().sin();

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);

    let result = expr.eval(&vars).expect("evaluation failed");
    assert_eq!(result, 0.0);
}

#[test]
fn test_contains_var() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let expr = x.clone() * x.clone() + y.clone();

    assert!(expr.contains_var("x"));
    assert!(expr.contains_var("y"));
    assert!(!expr.contains_var("z"));
}

#[test]
fn test_substitute_simple() {
    let x = Expr::var("x");
    let expr = x.clone() * 2.0;

    let five = Expr::constant(5.0);
    let substituted = expr.substitute("x", &five);

    let vars = HashMap::new();
    let result = substituted.eval(&vars).expect("evaluation failed");
    assert_eq!(result, 10.0);
}

#[test]
fn test_substitute_complex() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let expr = x.clone().pow(2.0) + y.clone();

    let y_expr = Expr::constant(3.0);
    let substituted = expr.substitute("y", &y_expr);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    let result = substituted.eval(&vars).expect("evaluation failed");
    assert_eq!(result, 7.0); // 4 + 3 = 7
}

#[test]
fn test_error_missing_variable() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let expr = x + y;

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);
    // y is missing

    let result = expr.eval(&vars);
    assert!(result.is_err());
}

#[test]
fn test_error_division_by_zero() {
    let x = Expr::var("x");
    let expr = x / 0.0;

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);

    let result = expr.eval(&vars);
    assert!(result.is_err());
}

#[test]
fn test_error_negative_sqrt() {
    let x = Expr::var("x");
    let expr = x.sqrt();

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), -1.0);

    let result = expr.eval(&vars);
    assert!(result.is_err());
}

#[test]
fn test_error_negative_ln() {
    let x = Expr::var("x");
    let expr = x.ln();

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), -1.0);

    let result = expr.eval(&vars);
    assert!(result.is_err());
}

#[test]
fn test_display() {
    let x = Expr::var("x");
    let expr = x.clone() * x.clone() + 1.0;

    let display_str = format!("{}", expr);
    assert!(display_str.contains("x"));
    assert!(display_str.contains("+"));
}

#[test]
fn test_latex_simple() {
    let x = Expr::var("x");
    let expr = x.pow(2.0);

    let latex = expr.to_latex();
    assert!(latex.contains("x"));
    assert!(latex.contains("^"));
}

#[test]
fn test_latex_fraction() {
    let x = Expr::var("x");
    let expr = x.clone() / (x.clone() + 1.0);

    let latex = expr.to_latex();
    assert!(latex.contains("\\frac"));
}

#[test]
fn test_latex_trig() {
    let x = Expr::var("x");
    let expr = x.sin();

    let latex = expr.to_latex();
    assert!(latex.contains("\\sin"));
}

#[test]
fn test_python_output() {
    let x = Expr::var("x");
    let expr = x.clone().pow(2.0);

    let python = expr.to_python();
    assert!(python.contains("**"));
}

#[test]
fn test_complex_expression() {
    let x = Expr::var("x");
    // f(x) = sin(x²) + exp(-x)
    let expr = (x.clone().pow(2.0)).sin() + (-x.clone()).exp();

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);

    let result = expr.eval(&vars).expect("evaluation failed");
    // sin(0) + exp(0) = 0 + 1 = 1
    assert_eq!(result, 1.0);
}

#[test]
fn test_nested_operations() {
    let x = Expr::var("x");
    let expr = ((x.clone() + 1.0) * 2.0).pow(3.0);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);

    let result = expr.eval(&vars).expect("evaluation failed");
    // ((1 + 1) * 2)³ = 4³ = 64
    assert_eq!(result, 64.0);
}
