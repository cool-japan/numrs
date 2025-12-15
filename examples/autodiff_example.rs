//! Automatic Differentiation Example
//!
//! This example demonstrates NumRS2's automatic differentiation capabilities,
//! including forward mode, reverse mode, and higher-order derivatives.
//!
//! Run with: cargo run --example autodiff_example

use numrs2::autodiff::*;
use numrs2::prelude::*;

fn main() {
    println!("=== NumRS2 Automatic Differentiation Examples ===\n");

    // Example 1: Forward Mode - Basic Derivatives
    println!("1. Forward Mode Automatic Differentiation");
    println!("-----------------------------------------");

    // f(x) = x^2 + 2x + 1, f'(x) = 2x + 2
    let x = Dual::variable(3.0);
    let result = x.pow(2.0) + x * Dual::constant(2.0) + Dual::constant(1.0);

    println!("f(x) = x² + 2x + 1 at x = 3.0:");
    println!("  f(3) = {}", result.value());
    println!("  f'(3) = {} (expected: 8.0)\n", result.deriv());

    // Example 2: Forward Mode - Transcendental Functions
    println!("2. Forward Mode with Transcendental Functions");
    println!("--------------------------------------------");

    // f(x) = e^x * sin(x), at x = 1.0
    let x = Dual::variable(1.0);
    let result = x.exp() * x.sin();

    println!("f(x) = e^x * sin(x) at x = 1.0:");
    println!("  f(1) = {:.6}", result.value());
    println!("  f'(1) = {:.6}\n", result.deriv());

    // Example 3: Forward Mode - Neural Network Activation
    println!("3. Forward Mode with Sigmoid Activation");
    println!("---------------------------------------");

    let x = Dual::variable(0.0);
    let sigmoid = x.sigmoid();

    println!("sigmoid(x) at x = 0.0:");
    println!("  σ(0) = {} (expected: 0.5)", sigmoid.value());
    println!("  σ'(0) = {} (expected: 0.25)\n", sigmoid.deriv());

    // Example 4: Reverse Mode - Gradient Computation
    println!("4. Reverse Mode Automatic Differentiation");
    println!("-----------------------------------------");

    let mut tape = Tape::new();

    // Variables
    let x = tape.var(2.0);
    let y = tape.var(3.0);

    // f(x, y) = x² + xy + y²
    let x_squared = tape.mul(x, x);
    let xy = tape.mul(x, y);
    let y_squared = tape.mul(y, y);
    let sum1 = tape.add(x_squared, xy);
    let f = tape.add(sum1, y_squared);

    // Compute gradients
    tape.backward(f);

    println!("f(x, y) = x² + xy + y² at (x=2, y=3):");
    println!("  f(2, 3) = {}", tape.value(f));
    println!("  ∂f/∂x = {} (expected: 2x + y = 7)", tape.grad(x));
    println!("  ∂f/∂y = {} (expected: x + 2y = 8)\n", tape.grad(y));

    // Example 5: Reverse Mode - Polynomial Loss
    println!("5. Reverse Mode for Polynomial Loss");
    println!("------------------------------------");

    let mut tape = Tape::new();

    // Polynomial: output = w*x^2 + b
    let x = tape.var(1.5);
    let w = tape.var(0.5);
    let b = tape.var(-0.2);

    let x_squared = tape.mul(x, x);
    let wx = tape.mul(w, x_squared);
    let output = tape.add(wx, b);

    // Loss = (output - target)²
    let target = tape.var(0.8);
    let diff = tape.sub(output, target);
    let loss = tape.mul(diff, diff);

    tape.backward(loss);

    println!("Function: output = w*x² + b");
    println!("Loss: (output - target)²");
    println!(
        "Values: w={}, x={}, b={}, target={}",
        tape.value(w),
        tape.value(x),
        tape.value(b),
        tape.value(target)
    );
    println!("  output = {:.4}", tape.value(output));
    println!("  loss = {:.6}", tape.value(loss));
    println!("Gradients:");
    println!("  ∂loss/∂w = {:.6}", tape.grad(w));
    println!("  ∂loss/∂b = {:.6}\n", tape.grad(b));

    // Example 6: Higher-Order Derivatives - Hessian
    println!("6. Higher-Order Derivatives (Hessian)");
    println!("-------------------------------------");

    // f(x, y) = x² + xy + y², compute Hessian
    let point = Array::from_vec(vec![2.0, 3.0]);
    let f_hessian = |x: &[f64]| x[0] * x[0] + x[0] * x[1] + x[1] * x[1];

    let hess = hessian(f_hessian, &point).unwrap();
    let hess_vec = hess.to_vec();

    println!("Hessian of f(x, y) = x² + xy + y² at (2, 3):");
    println!("  H = [[{:.1}, {:.1}],", hess_vec[0], hess_vec[1]);
    println!("       [{:.1}, {:.1}]]", hess_vec[2], hess_vec[3]);
    println!("  Expected: [[2, 1], [1, 2]]\n");

    // Example 7: Taylor Series Expansion
    println!("7. Taylor Series Expansion");
    println!("--------------------------");

    // Taylor series of sin(x) around x₀ = 0
    let f_taylor = |x: Dual<f64>| x.sin();
    let x0 = 0.0;
    let order = 5;

    let coeffs = taylor_series(f_taylor, x0, order);

    println!("Taylor series of sin(x) around x₀ = 0:");
    print!("  sin(x) ≈ ");
    for (i, &coeff) in coeffs.iter().enumerate() {
        if i > 0 && coeff >= 0.0 {
            print!("+ ");
        }
        if coeff.abs() > 1e-10 {
            print!("{:.6}x^{} ", coeff, i);
        }
    }
    println!("\n");

    // Example 8: Gradient Descent Optimization
    println!("8. Gradient Descent with Automatic Differentiation");
    println!("--------------------------------------------------");

    // Minimize f(x, y) = (x-1)² + (y-2)²
    let mut x = 0.0;
    let mut y = 0.0;
    let learning_rate = 0.1;

    println!("Minimizing f(x, y) = (x-1)² + (y-2)²");
    println!("Starting point: ({:.2}, {:.2})", x, y);

    for iteration in 0..10 {
        let mut tape = Tape::new();

        let x_var = tape.var(x);
        let y_var = tape.var(y);
        let one = tape.var(1.0);
        let two = tape.var(2.0);

        // f = (x-1)² + (y-2)²
        let x_diff = tape.sub(x_var, one);
        let y_diff = tape.sub(y_var, two);
        let x_sq = tape.mul(x_diff, x_diff);
        let y_sq = tape.mul(y_diff, y_diff);
        let loss = tape.add(x_sq, y_sq);

        tape.backward(loss);

        // Gradient descent update
        x -= learning_rate * tape.grad(x_var);
        y -= learning_rate * tape.grad(y_var);

        if iteration % 2 == 0 {
            println!(
                "  Iter {}: ({:.4}, {:.4}), loss = {:.6}",
                iteration,
                x,
                y,
                tape.value(loss)
            );
        }
    }

    println!(
        "  Final point: ({:.4}, {:.4}) (expected: (1.0, 2.0))\n",
        x, y
    );

    // Example 9: Jacobian Computation
    println!("9. Jacobian Matrix Computation");
    println!("------------------------------");

    // f: R² → R²
    // f₁(x, y) = x² + y
    // f₂(x, y) = xy
    let f_jacobian = |x: &Array<Dual<f64>>| {
        let x_vec = x.to_vec();
        let f1 = x_vec[0] * x_vec[0] + x_vec[1];
        let f2 = x_vec[0] * x_vec[1];
        Array::from_vec(vec![f1, f2])
    };

    let point = Array::from_vec(vec![2.0, 3.0]);
    let jac = jacobian(f_jacobian, &point).unwrap();
    let jac_vec = jac.to_vec();

    println!("Jacobian of f: R² → R² at (2, 3):");
    println!("  f₁(x,y) = x² + y");
    println!("  f₂(x,y) = xy");
    println!("  J = [[{:.1}, {:.1}],", jac_vec[0], jac_vec[1]);
    println!("       [{:.1}, {:.1}]]", jac_vec[2], jac_vec[3]);
    println!("  Expected: [[4, 1], [3, 2]]\n");

    println!("=== Examples Complete ===");
}
