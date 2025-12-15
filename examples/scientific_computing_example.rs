//! Scientific Computing Example for NumRS2
//!
//! This example demonstrates NumRS2's comprehensive scientific computing capabilities,
//! including numerical integration, ODE solving, PDE solving, signal processing,
//! and optimization techniques.

#![allow(clippy::needless_range_loop)]
//!
//! Run with: cargo run --example scientific_computing_example

use numrs2::integrate::{
    cumtrapz, dblquad, gauss_legendre, monte_carlo, monte_carlo_nd, quad, romberg, simps, trapz,
};
use numrs2::ode::{solve_ivp, solve_ivp_with_config, OdeConfig, OdeMethod};
use numrs2::pde::{
    solve_heat_1d, solve_heat_1d_crank_nicolson, solve_poisson_2d, BoundaryCondition,
};
use numrs2::prelude::*;
use std::f64::consts::PI;

fn main() {
    println!("=== NumRS2 Scientific Computing Examples ===\n");

    // Example 1: Numerical Integration
    numerical_integration_example();

    // Example 2: Ordinary Differential Equations
    ode_solving_example();

    // Example 3: Partial Differential Equations
    pde_solving_example();

    // Example 4: Signal Processing with FFT
    signal_processing_example();

    // Example 5: Optimization & Root Finding
    optimization_example();

    // Example 6: Monte Carlo Methods
    monte_carlo_example();

    // Example 7: Spectral Methods
    spectral_methods_example();

    println!("\n=== All Scientific Computing Examples Completed! ===");
}

/// Example 1: Numerical Integration Techniques
fn numerical_integration_example() {
    println!("1. Numerical Integration");
    println!("------------------------");

    // Integrate sin(x) from 0 to π (exact answer = 2)
    let f_sin = |x: f64| x.sin();

    // Trapezoidal rule
    let trap_result = trapz(f_sin, 0.0, PI, 100);
    println!(
        "  Trapezoidal (n=100): ∫sin(x)dx = {:.6} (exact: 2.0)",
        trap_result
    );

    // Simpson's rule
    let simp_result = simps(f_sin, 0.0, PI, 100);
    println!("  Simpson's (n=100):   ∫sin(x)dx = {:.6}", simp_result);

    // Romberg integration
    let romb_result = romberg(f_sin, 0.0, PI, 8);
    println!("  Romberg (order=8):   ∫sin(x)dx = {:.6}", romb_result);

    // Gauss-Legendre quadrature
    let gauss_result = gauss_legendre(f_sin, 0.0, PI, 5);
    println!("  Gauss-Legendre (5):  ∫sin(x)dx = {:.6}", gauss_result);

    // Adaptive quadrature
    match quad(f_sin, 0.0, PI) {
        Ok(result) => println!("  Adaptive quad:       ∫sin(x)dx = {:.6}", result),
        Err(e) => println!("  Adaptive quad error: {:?}", e),
    }

    // Double integral: ∫∫ x*y dA over [0,1]×[0,1] (exact = 0.25)
    let f_xy = |x: f64, y: f64| x * y;
    let dbl_result = dblquad(f_xy, 0.0, 1.0, 0.0, 1.0, 50, 50);
    println!(
        "\n  Double integral ∫∫xy dA = {:.6} (exact: 0.25)",
        dbl_result
    );

    // Cumulative trapezoidal integration
    let y_vals: Vec<f64> = (0..=10).map(|i| (i as f64 * 0.1 * PI).sin()).collect();
    let cum = cumtrapz(&y_vals, 0.1 * PI);
    println!(
        "  Cumulative trapz of sin: final = {:.6}",
        cum.last().unwrap_or(&0.0)
    );

    println!();
}

/// Example 2: Ordinary Differential Equations
fn ode_solving_example() {
    println!("2. ODE Solving");
    println!("--------------");

    // Example: Exponential decay dy/dt = -0.5*y, y(0) = 1
    // Exact solution: y(t) = exp(-0.5*t)
    let decay_rate = 0.5_f64;
    let f_decay = |_t: f64, y: &[f64]| vec![-decay_rate * y[0]];

    println!("  Solving dy/dt = -0.5*y, y(0) = 1");

    // RK4 method
    match solve_ivp(f_decay, (0.0, 4.0), &[1.0], OdeMethod::RK4) {
        Ok(result) => {
            let y_final = result.y.last().unwrap()[0];
            let exact = (-0.5_f64 * 4.0).exp();
            println!("    RK4:     y(4) = {:.6} (exact: {:.6})", y_final, exact);
        }
        Err(e) => println!("    RK4 error: {:?}", e),
    }

    // RK45 with adaptive stepping
    let config = OdeConfig {
        h0: 0.1,
        h_min: 1e-10,
        h_max: 0.5,
        atol: 1e-8,
        rtol: 1e-6,
        max_steps: 10000,
        t_eval: None,
    };
    match solve_ivp_with_config(f_decay, (0.0, 4.0), &[1.0], OdeMethod::RK45, &config) {
        Ok(result) => {
            let y_final = result.y.last().unwrap()[0];
            println!(
                "    RK45:    y(4) = {:.6} (steps: {})",
                y_final,
                result.t.len()
            );
        }
        Err(e) => println!("    RK45 error: {:?}", e),
    }

    // Harmonic oscillator: d²x/dt² = -ω²x
    // Written as system: dy[0]/dt = y[1], dy[1]/dt = -ω²*y[0]
    let omega = 2.0_f64;
    let f_harmonic = move |_t: f64, y: &[f64]| vec![y[1], -omega * omega * y[0]];

    println!("\n  Solving harmonic oscillator: x'' = -4x, x(0)=1, x'(0)=0");
    match solve_ivp(f_harmonic, (0.0, PI), &[1.0, 0.0], OdeMethod::RK4) {
        Ok(result) => {
            let x_final = result.y.last().unwrap()[0];
            let exact = (omega * PI).cos();
            println!("    RK4:     x(π) = {:.6} (exact: {:.6})", x_final, exact);
        }
        Err(e) => println!("    Harmonic oscillator error: {:?}", e),
    }

    // Lorenz system (chaotic attractor)
    let sigma = 10.0_f64;
    let rho = 28.0_f64;
    let beta = 8.0_f64 / 3.0_f64;
    let f_lorenz = move |_t: f64, y: &[f64]| {
        vec![
            sigma * (y[1] - y[0]),
            y[0] * (rho - y[2]) - y[1],
            y[0] * y[1] - beta * y[2],
        ]
    };

    println!("\n  Solving Lorenz system (chaotic)");
    match solve_ivp(f_lorenz, (0.0, 1.0), &[1.0, 1.0, 1.0], OdeMethod::RK4) {
        Ok(result) => {
            let y_final = result.y.last().unwrap();
            println!(
                "    At t=1: x={:.4}, y={:.4}, z={:.4}",
                y_final[0], y_final[1], y_final[2]
            );
        }
        Err(e) => println!("    Lorenz system error: {:?}", e),
    }

    println!();
}

/// Example 3: Partial Differential Equations
fn pde_solving_example() {
    println!("3. PDE Solving");
    println!("--------------");

    // 1D Heat equation: u_t = α * u_xx
    let nx = 50_usize;
    let nt = 100_usize;
    let alpha = 0.01_f64;
    let dx = 1.0 / (nx as f64 - 1.0);
    let dt = 0.0001_f64;

    // Initial condition: sin(π*x)
    let initial: Vec<f64> = (0..nx).map(|i| (PI * i as f64 * dx).sin()).collect();

    // Boundary conditions: u(0,t) = u(1,t) = 0
    let bc_left = BoundaryCondition::Dirichlet(0.0);
    let bc_right = BoundaryCondition::Dirichlet(0.0);

    println!("  1D Heat Equation (explicit FTCS):");
    match solve_heat_1d(&initial, alpha, dx, dt, nt, bc_left, bc_right) {
        Ok(result) => {
            // result.u is Vec<Vec<T>> - solution at each time step
            if let Some(final_u) = result.u.last() {
                let max_u = final_u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                println!("    Final max temperature: {:.6}", max_u);
            }
            let r = alpha * dt / (dx * dx);
            println!(
                "    Success: {} (CFL r={:.4}, stable if r≤0.5)",
                result.success, r
            );
        }
        Err(e) => println!("    Heat equation error: {:?}", e),
    }

    // 1D Heat with Crank-Nicolson (implicit, unconditionally stable)
    println!("\n  1D Heat Equation (Crank-Nicolson):");
    let dt_large = 0.001_f64; // Larger time step
    match solve_heat_1d_crank_nicolson(&initial, alpha, dx, dt_large, nt, bc_left, bc_right) {
        Ok(result) => {
            if let Some(final_u) = result.u.last() {
                let max_u = final_u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                println!("    Final max temperature: {:.6}", max_u);
            }
            println!("    Always stable (implicit scheme)");
        }
        Err(e) => println!("    Crank-Nicolson error: {:?}", e),
    }

    // 2D Poisson equation: ∇²u = f
    let nx_2d = 20_usize;
    let ny_2d = 20_usize;

    // Source term: uniform -1 (flattened 2D array)
    let source: Vec<f64> = vec![-1.0; nx_2d * ny_2d];

    println!("\n  2D Poisson Equation (Jacobi iteration):");
    match solve_poisson_2d(
        &source,
        nx_2d,
        ny_2d,
        dx,
        dx,
        BoundaryCondition::Dirichlet(0.0),
        1000,
        1e-6,
    ) {
        Ok((u, iterations, residual)) => {
            let center = u[ny_2d / 2 * nx_2d + nx_2d / 2];
            println!("    Center value: {:.6}", center);
            println!(
                "    Converged in {} iterations (residual: {:.2e})",
                iterations, residual
            );
        }
        Err(e) => println!("    Poisson error: {:?}", e),
    }

    println!();
}

/// Example 4: Signal Processing with FFT
fn signal_processing_example() {
    println!("4. Signal Processing");
    println!("--------------------");

    let n = 256_usize;
    let sample_rate = 1000.0_f64; // Hz
    let dt = 1.0 / sample_rate;

    // Generate composite signal: 50 Hz + 120 Hz
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 * dt;
            (2.0 * PI * 50.0 * t).sin() + 0.5 * (2.0 * PI * 120.0 * t).sin()
        })
        .collect();

    println!("  Signal: sin(2π·50t) + 0.5·sin(2π·120t)");
    println!("  Sample rate: {} Hz, {} samples", sample_rate, n);

    // Compute power spectral density using array operations
    let signal_arr = Array::from_vec(signal.clone());
    let mean = signal_arr.mean();
    let variance = signal_arr.var();
    println!("  Signal stats: mean={:.4}, variance={:.4}", mean, variance);

    // Find peaks by computing DFT magnitude for key frequencies
    let mut peak_freqs = Vec::new();
    for &freq in &[50.0_f64, 120.0_f64] {
        let k = (freq * n as f64 / sample_rate).round() as usize;
        let mut real = 0.0_f64;
        let mut imag = 0.0_f64;
        for (i, &s) in signal.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * i as f64 / n as f64;
            real += s * angle.cos();
            imag += s * angle.sin();
        }
        let magnitude = (real * real + imag * imag).sqrt() / n as f64 * 2.0;
        peak_freqs.push((freq, magnitude));
    }

    println!("  Detected peaks:");
    for (freq, mag) in peak_freqs {
        println!("    {:.0} Hz: amplitude = {:.4}", freq, mag);
    }

    // Windowing demonstration
    let hamming: Vec<f64> = (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        .collect();
    let windowed: Vec<f64> = signal
        .iter()
        .zip(hamming.iter())
        .map(|(s, w)| s * w)
        .collect();

    let windowed_arr = Array::from_vec(windowed);
    let windowed_energy: f64 = windowed_arr.to_vec().iter().map(|x| x * x).sum();
    println!("  Hamming windowed energy: {:.4}", windowed_energy);

    println!();
}

/// Example 5: Optimization & Root Finding
fn optimization_example() {
    println!("5. Optimization & Root Finding");
    println!("------------------------------");

    // Newton-Raphson for finding roots
    // Find root of f(x) = x³ - 2x - 5 near x=2
    // f'(x) = 3x² - 2
    let mut x = 2.0_f64;
    println!("  Newton-Raphson: finding root of x³ - 2x - 5 = 0");
    println!("    Initial guess: x = {:.4}", x);

    for i in 0..10 {
        let f = x * x * x - 2.0 * x - 5.0;
        let df = 3.0 * x * x - 2.0;
        if df.abs() < 1e-10 {
            break;
        }
        x -= f / df;
        if f.abs() < 1e-12 {
            println!("    Converged at iteration {}", i + 1);
            break;
        }
    }
    println!("    Root: x = {:.10}", x);
    println!(
        "    Verification: f({:.6}) = {:.2e}",
        x,
        x * x * x - 2.0 * x - 5.0
    );

    // Gradient descent for minimization
    // Minimize f(x,y) = (x-1)² + (y-2)²
    let mut xy = [0.0_f64, 0.0_f64];
    let lr = 0.1_f64;

    println!("\n  Gradient descent: minimize (x-1)² + (y-2)²");
    println!("    Initial: ({:.4}, {:.4})", xy[0], xy[1]);

    for _ in 0..100 {
        let grad_x = 2.0 * (xy[0] - 1.0);
        let grad_y = 2.0 * (xy[1] - 2.0);
        xy[0] -= lr * grad_x;
        xy[1] -= lr * grad_y;
    }
    println!("    Final: ({:.6}, {:.6})", xy[0], xy[1]);
    println!(
        "    Minimum value: {:.2e}",
        (xy[0] - 1.0).powi(2) + (xy[1] - 2.0).powi(2)
    );

    // Golden section search for 1D minimization
    // Minimize f(x) = (x-3)² on [0, 5]
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut a = 0.0_f64;
    let mut b = 5.0_f64;
    let tol = 1e-8_f64;

    println!("\n  Golden section search: minimize (x-3)² on [0,5]");

    while (b - a) > tol {
        let c = b - (b - a) / phi;
        let d = a + (b - a) / phi;
        let fc = (c - 3.0) * (c - 3.0);
        let fd = (d - 3.0) * (d - 3.0);
        if fc < fd {
            b = d;
        } else {
            a = c;
        }
    }
    let x_min = (a + b) / 2.0;
    println!("    Minimum at x = {:.10}", x_min);

    // Bisection method for root finding
    // Find root of cos(x) - x = 0 in [0, 1]
    let mut a = 0.0_f64;
    let mut b = 1.0_f64;
    let f = |x: f64| x.cos() - x;

    println!("\n  Bisection: find root of cos(x) - x = 0");

    for _ in 0..50 {
        let c = (a + b) / 2.0;
        if f(c).abs() < 1e-12 || (b - a) / 2.0 < 1e-12 {
            println!("    Root: x = {:.10}", c);
            println!("    Verification: cos({:.6}) - {:.6} = {:.2e}", c, c, f(c));
            break;
        }
        if f(a) * f(c) < 0.0 {
            b = c;
        } else {
            a = c;
        }
    }

    println!();
}

/// Example 6: Monte Carlo Methods
fn monte_carlo_example() {
    println!("6. Monte Carlo Methods");
    println!("----------------------");

    // Monte Carlo integration: ∫₀¹ √(1-x²) dx = π/4
    let f_quarter_circle = |x: f64| (1.0 - x * x).sqrt();

    println!("  1D Monte Carlo: ∫₀¹ √(1-x²) dx = π/4");
    for n in [1000_usize, 10000, 100000] {
        let result = monte_carlo(f_quarter_circle, 0.0, 1.0, n);
        let exact = PI / 4.0;
        let error = (result.value - exact).abs();
        println!(
            "    n={:6}: {:.6} (error: {:.2e}, std_err: {:.2e})",
            n, result.value, error, result.error
        );
    }

    // Multi-dimensional Monte Carlo: volume of unit sphere
    // ∫∫∫ (x² + y² + z² ≤ 1) dV = 4π/3
    let f_sphere = |coords: &[f64]| {
        let r_sq = coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2];
        if r_sq <= 1.0 {
            1.0
        } else {
            0.0
        }
    };

    // Integrate over [-1,1]³, multiply by 8 for full cube
    let bounds = vec![(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)];

    println!("\n  3D Monte Carlo: Volume of unit sphere = 4π/3");
    for n in [10000_usize, 100000, 1000000] {
        let result = monte_carlo_nd(f_sphere, &bounds, n);
        let exact = 4.0 * PI / 3.0;
        let error = (result.value - exact).abs();
        println!(
            "    n={:7}: {:.6} (error: {:.4}, expected: {:.6})",
            n, result.value, error, exact
        );
    }

    // Monte Carlo estimation of E[X²] for standard normal
    // E[X²] = 1 for N(0,1)
    println!("\n  Monte Carlo expectation: E[X²] for N(0,1)");
    let n_samples = 100000_usize;

    // Use Box-Muller transform to generate normal samples
    let mut sum_sq = 0.0_f64;
    let mut count = 0_usize;

    // Simple LCG for reproducibility
    let mut seed = 12345_u64;
    let lcg = |s: &mut u64| {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s as f64) / (u64::MAX as f64)
    };

    for _ in 0..n_samples / 2 {
        let u1 = lcg(&mut seed);
        let u2 = lcg(&mut seed);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        let z1 = r * theta.cos();
        let z2 = r * theta.sin();
        sum_sq += z1 * z1 + z2 * z2;
        count += 2;
    }

    let estimate = sum_sq / count as f64;
    println!("    E[X²] estimate: {:.6} (exact: 1.0)", estimate);

    println!();
}

/// Example 7: Spectral Methods
fn spectral_methods_example() {
    println!("7. Spectral Methods");
    println!("-------------------");

    // Chebyshev polynomial approximation
    let n = 10_usize;

    // Chebyshev nodes on [-1, 1]
    let nodes: Vec<f64> = (0..=n).map(|k| (PI * k as f64 / n as f64).cos()).collect();

    // Approximate f(x) = exp(x) using Chebyshev interpolation
    let f = |x: f64| x.exp();
    let f_values: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();

    // Compute Chebyshev coefficients using DCT-like formula
    let mut coeffs = vec![0.0_f64; n + 1];
    for j in 0..=n {
        let mut sum = 0.0;
        for k in 0..=n {
            let weight = if k == 0 || k == n { 0.5 } else { 1.0 };
            sum += weight * f_values[k] * (PI * j as f64 * k as f64 / n as f64).cos();
        }
        coeffs[j] = 2.0 * sum / n as f64;
        if j == 0 {
            coeffs[j] /= 2.0;
        }
    }

    println!("  Chebyshev approximation of exp(x) on [-1, 1]");
    println!(
        "  First 5 coefficients: {:.6}, {:.6}, {:.6}, {:.6}, {:.6}",
        coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]
    );

    // Evaluate approximation at test points
    let test_points = [-0.5_f64, 0.0, 0.5];
    println!("  Approximation accuracy:");
    for &x in &test_points {
        // Clenshaw algorithm for evaluating Chebyshev series
        let mut b1 = 0.0_f64;
        let mut b2 = 0.0_f64;
        for k in (1..=n).rev() {
            let b_new = coeffs[k] + 2.0 * x * b1 - b2;
            b2 = b1;
            b1 = b_new;
        }
        let approx = coeffs[0] + x * b1 - b2;
        let exact = f(x);
        println!(
            "    f({:.1}) = {:.10} (approx), {:.10} (exact), error = {:.2e}",
            x,
            approx,
            exact,
            (approx - exact).abs()
        );
    }

    // Legendre-Gauss quadrature nodes and weights for n=5
    println!("\n  Legendre-Gauss quadrature (n=5):");

    // Known Gauss-Legendre nodes and weights for n=5
    let gl_nodes = [
        -0.906_179_845_938_664,
        -0.5384693101056831,
        0.0,
        0.5384693101056831,
        0.906_179_845_938_664,
    ];
    let gl_weights = [
        0.2369268850561891,
        0.4786286704993665,
        0.5688888888888889,
        0.4786286704993665,
        0.2369268850561891,
    ];

    // Integrate x⁴ from -1 to 1 (exact = 2/5)
    let f_x4 = |x: f64| x.powi(4);
    let mut gl_result = 0.0_f64;
    for (&node, &weight) in gl_nodes.iter().zip(gl_weights.iter()) {
        gl_result += weight * f_x4(node);
    }
    println!("    ∫₋₁¹ x⁴ dx = {:.10} (exact: 0.4)", gl_result);

    // Spectral differentiation demonstration
    println!("\n  Spectral differentiation:");

    // Simple finite difference comparison for d/dx[sin(x)] = cos(x)
    let h = 0.01_f64;
    let x_test = 0.5_f64;
    let fd_deriv = (f64::sin(x_test + h) - f64::sin(x_test - h)) / (2.0 * h);
    let spectral_approx = x_test.cos(); // exact derivative
    println!("    d/dx[sin({:.1})] at x={:.1}:", x_test, x_test);
    println!("      Finite difference (h={:.2}): {:.10}", h, fd_deriv);
    println!("      Exact derivative:            {:.10}", spectral_approx);
    println!("      FD error: {:.2e}", (fd_deriv - spectral_approx).abs());

    println!();
}
