//! Symbolic Mathematics Example for NumRS2
//!
//! This example demonstrates the symbolic computation capabilities of NumRS2,
//! including expression manipulation, differentiation, simplification, and
//! symbolic linear algebra.

use numrs2::symbolic::*;
use std::collections::HashMap;

fn main() {
    println!("=== NumRS2 Symbolic Mathematics Examples ===\n");

    // Example 1: Basic expression creation and evaluation
    basic_expressions();

    // Example 2: Symbolic differentiation
    symbolic_differentiation();

    // Example 3: Gradient computation
    gradient_computation();

    // Example 4: Expression simplification
    expression_simplification();

    // Example 5: Expression expansion
    expression_expansion();

    // Example 6: LaTeX output
    latex_output();

    // Example 7: Symbolic linear algebra
    symbolic_linear_algebra();

    // Example 8: Optimization example
    optimization_example();

    // Example 9: Chain rule demonstration
    chain_rule_demo();

    // Example 10: Taylor series
    taylor_series();
}

/// Example 1: Basic expression creation and evaluation
fn basic_expressions() {
    println!("--- Example 1: Basic Expressions ---");

    let x = Expr::var("x");
    let y = Expr::var("y");

    // Create expression: f(x, y) = x² + 2xy + y²
    let expr = x.clone().pow(2.0) + x.clone() * y.clone() * 2.0 + y.clone().pow(2.0);

    println!("Expression: {}", expr);

    // Evaluate at x = 3, y = 4
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 3.0);
    vars.insert("y".to_string(), 4.0);

    match expr.eval(&vars) {
        Ok(result) => {
            println!("f(3, 4) = {}", result); // (3 + 4)² = 49
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!();
}

/// Example 2: Symbolic differentiation
fn symbolic_differentiation() {
    println!("--- Example 2: Symbolic Differentiation ---");

    let x = Expr::var("x");

    // f(x) = x³ - 2x² + 3x - 5
    let f = x.clone().pow(3.0) - x.clone().pow(2.0) * 2.0 + x.clone() * 3.0 - 5.0;

    println!("f(x) = {}", f);

    // Compute derivative
    match differentiate(&f, "x") {
        Ok(df) => {
            println!("f'(x) = {}", df);

            // Simplify
            let simplified = simplify(&df);
            println!("f'(x) simplified = {}", simplified);

            // Evaluate at x = 2
            let mut vars = HashMap::new();
            vars.insert("x".to_string(), 2.0);

            match simplified.eval(&vars) {
                Ok(result) => {
                    println!("f'(2) = {}", result); // 3*4 - 4*2 + 3 = 12 - 8 + 3 = 7
                }
                Err(e) => println!("Error: {:?}", e),
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!();
}

/// Example 3: Gradient computation
fn gradient_computation() {
    println!("--- Example 3: Gradient Computation ---");

    let x = Expr::var("x");
    let y = Expr::var("y");

    // f(x, y) = x² + y² (distance squared from origin)
    let f = x.clone().pow(2.0) + y.clone().pow(2.0);

    println!("f(x, y) = {}", f);

    // Compute gradient: ∇f = [∂f/∂x, ∂f/∂y]
    match gradient(&f, &["x", "y"]) {
        Ok(grad) => {
            println!("∇f = [{}, {}]", grad[0], grad[1]);

            // Simplify each component
            let grad_simplified: Vec<Expr> = grad.iter().map(simplify).collect();
            println!(
                "∇f simplified = [{}, {}]",
                grad_simplified[0], grad_simplified[1]
            );

            // Evaluate at (3, 4)
            let mut vars = HashMap::new();
            vars.insert("x".to_string(), 3.0);
            vars.insert("y".to_string(), 4.0);

            match (
                grad_simplified[0].eval(&vars),
                grad_simplified[1].eval(&vars),
            ) {
                (Ok(dx), Ok(dy)) => {
                    println!("∇f(3, 4) = [{}, {}]", dx, dy); // [6, 8]
                }
                _ => println!("Error evaluating gradient"),
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!();
}

/// Example 4: Expression simplification
fn expression_simplification() {
    println!("--- Example 4: Expression Simplification ---");

    let x = Expr::var("x");

    // Create a complex expression with redundant operations
    let expr = (x.clone() + 0.0) * (x.clone() + 1.0) + (x.clone() * 0.0);

    println!("Original: {}", expr);

    let simplified = simplify(&expr);
    println!("Simplified: {}", simplified);

    // Another example with double negation
    let expr2 = -(-x.clone());
    println!("\nOriginal: {}", expr2);
    println!("Simplified: {}", simplify(&expr2));

    println!();
}

/// Example 5: Expression expansion
fn expression_expansion() {
    println!("--- Example 5: Expression Expansion ---");

    let x = Expr::var("x");

    // (x + 1)(x + 2) = x² + 3x + 2
    let expr = (x.clone() + 1.0) * (x.clone() + 2.0);

    println!("Original: {}", expr);

    let expanded = expand(&expr);
    println!("Expanded: {}", expanded);

    let simplified = simplify(&expanded);
    println!("Simplified: {}", simplified);

    // Verify by evaluation
    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 1.0);

    match (expr.eval(&vars), simplified.eval(&vars)) {
        (Ok(v1), Ok(v2)) => {
            println!("\nVerification at x=1:");
            println!("Original value: {}", v1);
            println!("Expanded value: {}", v2);
            println!("Match: {}", (v1 - v2).abs() < 1e-10);
        }
        _ => println!("Error in evaluation"),
    }

    println!();
}

/// Example 6: LaTeX output
fn latex_output() {
    println!("--- Example 6: LaTeX Output ---");

    let x = Expr::var("x");

    // Quadratic formula
    let expr = (x.clone().pow(2.0) + x.clone() * 2.0 + 1.0) / (x.clone() - 1.0);

    println!("Expression: {}", expr);
    println!("LaTeX: {}", expr.to_latex());
    println!("Python: {}", expr.to_python());

    // Trigonometric expression
    let trig_expr = x.clone().sin().pow(2.0) + x.clone().cos().pow(2.0);
    println!("\nExpression: {}", trig_expr);
    println!("LaTeX: {}", trig_expr.to_latex());

    println!();
}

/// Example 7: Symbolic linear algebra
fn symbolic_linear_algebra() {
    println!("--- Example 7: Symbolic Linear Algebra ---");

    let x = Expr::var("x");

    // Create a symbolic matrix: [[x, 1], [0, x]]
    let mat_data = vec![
        vec![x.clone(), Expr::constant(1.0)],
        vec![Expr::constant(0.0), x.clone()],
    ];

    match SymbolicMatrix::from_vec(mat_data) {
        Ok(mat) => {
            println!("Matrix:");
            for i in 0..mat.nrows() {
                print!("[");
                for j in 0..mat.ncols() {
                    if let Some(elem) = mat.get(i, j) {
                        print!("{}", elem);
                        if j < mat.ncols() - 1 {
                            print!(", ");
                        }
                    }
                }
                println!("]");
            }

            // Compute determinant
            match determinant(&mat) {
                Ok(det) => {
                    println!("\nDeterminant: {}", det);
                    let det_simplified = simplify(&det);
                    println!("Simplified: {}", det_simplified);

                    // Evaluate at x = 3
                    let mut vars = HashMap::new();
                    vars.insert("x".to_string(), 3.0);

                    match det_simplified.eval(&vars) {
                        Ok(result) => {
                            println!("det(x=3) = {}", result); // 3 * 3 = 9
                        }
                        Err(e) => println!("Error: {:?}", e),
                    }
                }
                Err(e) => println!("Error computing determinant: {:?}", e),
            }

            // Compute trace
            match trace(&mat) {
                Ok(tr) => {
                    println!("\nTrace: {}", tr);
                    let tr_simplified = simplify(&tr);
                    println!("Simplified: {}", tr_simplified);
                }
                Err(e) => println!("Error computing trace: {:?}", e),
            }
        }
        Err(e) => println!("Error creating matrix: {:?}", e),
    }

    println!();
}

/// Example 8: Optimization example
fn optimization_example() {
    println!("--- Example 8: Optimization Example ---");

    let x = Expr::var("x");

    // Find minimum of f(x) = (x - 3)² + 2
    let f = (x.clone() - 3.0).pow(2.0) + 2.0;

    println!("Objective function: f(x) = {}", f);

    // Compute derivative
    match differentiate(&f, "x") {
        Ok(df) => {
            let df_simplified = simplify(&df);
            println!("f'(x) = {}", df_simplified);

            // The minimum occurs where f'(x) = 0
            // For this simple case, we know x = 3 is the minimum

            let mut vars = HashMap::new();
            vars.insert("x".to_string(), 3.0);

            match (f.eval(&vars), df_simplified.eval(&vars)) {
                (Ok(f_val), Ok(df_val)) => {
                    println!("\nAt x = 3:");
                    println!("f(3) = {}", f_val); // 2
                    println!("f'(3) = {}", df_val); // 0 (critical point)
                }
                _ => println!("Error in evaluation"),
            }

            // Check second derivative to confirm it's a minimum
            match differentiate(&df, "x") {
                Ok(d2f) => {
                    let d2f_simplified = simplify(&d2f);
                    println!("f''(x) = {}", d2f_simplified);

                    match d2f_simplified.eval(&vars) {
                        Ok(d2f_val) => {
                            if d2f_val > 0.0 {
                                println!("f''(3) = {} > 0, so x=3 is a minimum", d2f_val);
                            }
                        }
                        Err(e) => println!("Error: {:?}", e),
                    }
                }
                Err(e) => println!("Error computing second derivative: {:?}", e),
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!();
}

/// Example 9: Chain rule demonstration
fn chain_rule_demo() {
    println!("--- Example 9: Chain Rule Demonstration ---");

    let x = Expr::var("x");

    // f(x) = sin(x²)
    let f = (x.clone().pow(2.0)).sin();

    println!("f(x) = {}", f);

    // Compute derivative using chain rule: f'(x) = cos(x²) * 2x
    match differentiate(&f, "x") {
        Ok(df) => {
            println!("f'(x) = {}", df);

            // Evaluate at x = 0
            let mut vars = HashMap::new();
            vars.insert("x".to_string(), 0.0);

            match df.eval(&vars) {
                Ok(result) => {
                    println!("f'(0) = {}", result); // cos(0) * 0 = 0
                }
                Err(e) => println!("Error: {:?}", e),
            }

            // Evaluate at x = 1
            vars.insert("x".to_string(), 1.0);
            match df.eval(&vars) {
                Ok(result) => {
                    println!("f'(1) = {}", result); // cos(1) * 2
                    let expected = 1.0_f64.cos() * 2.0;
                    println!("Expected: {}", expected);
                    println!("Match: {}", (result - expected).abs() < 1e-10);
                }
                Err(e) => println!("Error: {:?}", e),
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }

    println!();
}

/// Example 10: Taylor series approximation
fn taylor_series() {
    println!("--- Example 10: Taylor Series Approximation ---");

    let x = Expr::var("x");

    // Compute Taylor series of exp(x) around x = 0
    // exp(x) ≈ 1 + x + x²/2 + x³/6 + ...

    let f = x.clone().exp();
    println!("f(x) = {}", f);

    let mut vars = HashMap::new();
    vars.insert("x".to_string(), 0.0);

    // Compute derivatives at x = 0
    let mut current = f.clone();
    let mut coeffs = vec![];

    for n in 0..5 {
        match current.eval(&vars) {
            Ok(coeff) => {
                coeffs.push(coeff);
                println!("f^({})({}) = {}", "′".repeat(n), 0, coeff);
            }
            Err(e) => {
                println!("Error: {:?}", e);
                break;
            }
        }

        match differentiate(&current, "x") {
            Ok(deriv) => current = deriv,
            Err(e) => {
                println!("Error: {:?}", e);
                break;
            }
        }
    }

    // Build Taylor polynomial: 1 + x + x²/2 + x³/6 + x⁴/24
    println!("\nTaylor series approximation:");
    println!("exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + ...");

    // Test approximation at x = 0.5
    let test_x = 0.5_f64;
    let actual = test_x.exp();

    let mut taylor_approx = 0.0;
    let mut factorial = 1.0;
    let mut x_power = 1.0;

    for (n, coeff) in coeffs.iter().enumerate() {
        if n > 0 {
            factorial *= n as f64;
            x_power *= test_x;
        }
        taylor_approx += coeff * x_power / factorial;
    }

    println!("\nAt x = {}:", test_x);
    println!("Actual exp({}) = {}", test_x, actual);
    println!("Taylor approximation = {}", taylor_approx);
    println!("Error = {}", (actual - taylor_approx).abs());

    println!();
}
