//! Tests for expression simplification

use numrs2::symbolic::*;
use std::collections::HashMap;

#[test]
fn test_simplify_addition_identity() {
    let x = Expr::var("x");

    // x + 0 = x
    let expr = x.clone() + 0.0;
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);

    // 0 + x = x
    let expr = 0.0 + x.clone();
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_constant_folding_add() {
    // 2 + 3 = 5
    let expr = Expr::constant(2.0) + Expr::constant(3.0);
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(5.0));
}

#[test]
fn test_simplify_multiplication_identity() {
    let x = Expr::var("x");

    // x * 1 = x
    let expr = x.clone() * 1.0;
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);

    // 1 * x = x
    let expr = 1.0 * x.clone();
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_multiplication_zero() {
    let x = Expr::var("x");

    // x * 0 = 0
    let expr = x.clone() * 0.0;
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(0.0));

    // 0 * x = 0
    let expr = 0.0 * x.clone();
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(0.0));
}

#[test]
fn test_simplify_constant_folding_mul() {
    // 2 * 3 = 6
    let expr = Expr::constant(2.0) * Expr::constant(3.0);
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(6.0));
}

#[test]
fn test_simplify_subtraction_identity() {
    let x = Expr::var("x");

    // x - 0 = x
    let expr = x.clone() - 0.0;
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_subtraction_self() {
    let x = Expr::var("x");

    // x - x = 0
    let expr = x.clone() - x.clone();
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(0.0));
}

#[test]
fn test_simplify_division_identity() {
    let x = Expr::var("x");

    // x / 1 = x
    let expr = x.clone() / 1.0;
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_division_self() {
    let x = Expr::var("x");

    // x / x = 1
    let expr = x.clone() / x.clone();
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(1.0));
}

#[test]
fn test_simplify_zero_division() {
    let x = Expr::var("x");

    // 0 / x = 0
    let expr = 0.0 / x.clone();
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(0.0));
}

#[test]
fn test_simplify_power_zero() {
    let x = Expr::var("x");

    // x^0 = 1
    let expr = x.pow(0.0);
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(1.0));
}

#[test]
fn test_simplify_power_one() {
    let x = Expr::var("x");

    // x^1 = x
    let expr = x.clone().pow(1.0);
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_constant_folding_pow() {
    // 2^3 = 8
    let expr = Expr::constant(2.0).pow(3.0);
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(8.0));
}

#[test]
fn test_simplify_double_negation() {
    let x = Expr::var("x");

    // -(-x) = x
    let expr = -(-x.clone());
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_negation_zero() {
    // -0 = 0
    let expr = -Expr::constant(0.0);
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(0.0));
}

#[test]
fn test_simplify_sin_zero() {
    // sin(0) = 0
    let expr = Expr::constant(0.0).sin();
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(0.0));
}

#[test]
fn test_simplify_cos_zero() {
    // cos(0) = 1
    let expr = Expr::constant(0.0).cos();
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(1.0));
}

#[test]
fn test_simplify_exp_zero() {
    // exp(0) = 1
    let expr = Expr::constant(0.0).exp();
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(1.0));
}

#[test]
fn test_simplify_ln_one() {
    // ln(1) = 0
    let expr = Expr::constant(1.0).ln();
    let simplified = simplify(&expr);
    assert_eq!(simplified, Expr::constant(0.0));
}

#[test]
fn test_simplify_exp_ln_inverse() {
    let x = Expr::var("x");

    // exp(ln(x)) = x
    let expr = x.clone().ln().exp();
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_ln_exp_inverse() {
    let x = Expr::var("x");

    // ln(exp(x)) = x
    let expr = x.clone().exp().ln();
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);
}

#[test]
fn test_simplify_nested() {
    let x = Expr::var("x");

    // ((x + 0) * 1) + 0 = x
    let expr = ((x.clone() + 0.0) * 1.0) + 0.0;
    let simplified = simplify(&expr);
    assert_eq!(simplified, x);
}

#[test]
fn test_expand_distributive_simple() {
    let x = Expr::var("x");

    // 2 * (x + 1) = 2x + 2
    let expr = 2.0 * (x.clone() + 1.0);
    let expanded = expand(&expr);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    let original_val = expr.eval(&vars).expect("eval failed");
    let expanded_val = expanded.eval(&vars).expect("eval failed");

    assert_eq!(original_val, expanded_val);
}

#[test]
fn test_expand_binomial() {
    let x = Expr::var("x");

    // (x + 1) * (x + 2) = x² + 3x + 2
    let expr = (x.clone() + 1.0) * (x.clone() + 2.0);
    let expanded = expand(&expr);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);

    let original_val = expr.eval(&vars).expect("eval failed");
    let expanded_val = expanded.eval(&vars).expect("eval failed");

    assert_eq!(original_val, expanded_val);
    // At x=1: (1+1)*(1+2) = 2*3 = 6
    assert_eq!(original_val, 6.0);
}

#[test]
fn test_expand_power() {
    let x = Expr::var("x");

    // (x + 1)² = x² + 2x + 1
    let expr = (x.clone() + 1.0).pow(2.0);
    let expanded = expand(&expr);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    let original_val = expr.eval(&vars).expect("eval failed");
    let expanded_val = expanded.eval(&vars).expect("eval failed");

    assert_eq!(original_val, expanded_val);
    // At x=2: (2+1)² = 9
    assert_eq!(original_val, 9.0);
}

#[test]
fn test_simplify_complex_expression() {
    let x = Expr::var("x");

    // (x * 0) + (x * 1) + (0 * x) = 0 + x + 0 = x
    let expr = (x.clone() * 0.0) + (x.clone() * 1.0) + (0.0 * x.clone());
    let simplified = simplify(&expr);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 5.0);

    let result = simplified.eval(&vars).expect("eval failed");
    assert_eq!(result, 5.0);
}

#[test]
fn test_simplify_after_differentiation() {
    let x = Expr::var("x");
    let expr = x.clone().pow(2.0) + x.clone() * 0.0;

    let derivative = differentiate(&expr, "x").expect("differentiation failed");
    let simplified = simplify(&derivative);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);

    let result = simplified.eval(&vars).expect("eval failed");

    // d/dx(x² + 0) = 2x, at x=3: 6
    assert_eq!(result, 6.0);
}

#[test]
fn test_expand_then_simplify() {
    let x = Expr::var("x");

    // (x + 0) * (x + 1)
    let expr = (x.clone() + 0.0) * (x.clone() + 1.0);
    let expanded = expand(&expr);
    let simplified = simplify(&expanded);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 2.0);

    let result = simplified.eval(&vars).expect("eval failed");
    // x * (x + 1) = x² + x, at x=2: 4 + 2 = 6
    assert_eq!(result, 6.0);
}
