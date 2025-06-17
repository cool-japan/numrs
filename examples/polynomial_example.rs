#![allow(deprecated)]

use numrs2::math::linspace;
use numrs2::prelude::*;

fn main() {
    println!("NumRS Polynomial Functions Examples");
    println!("===================================\n");

    // 1. Basic polynomial operations
    println!("1. Basic Polynomial Operations");
    println!("------------------------------");

    // Create polynomials: p(x) = 2x^2 + 3x + 1 and q(x) = x + 2
    let p = Polynomial::new(vec![2.0, 3.0, 1.0]);
    let q = Polynomial::new(vec![1.0, 2.0]);

    println!("p(x) coefficients: {:?}", p.coefficients());
    println!("q(x) coefficients: {:?}", q.coefficients());

    println!("\nDegree of p(x): {}", p.degree());
    println!("Degree of q(x): {}", q.degree());

    // Evaluate polynomials
    println!("\nEvaluating polynomials:");
    println!("p(0) = {}", p.evaluate(0.0));
    println!("p(1) = {}", p.evaluate(1.0));
    println!("p(2) = {}", p.evaluate(2.0));

    // Arithmetic operations
    let sum = p.clone() + q.clone();
    let product = p.clone() * q.clone();

    println!("\nPolynomial arithmetic:");
    println!("(p + q) coefficients: {:?}", sum.coefficients());
    println!("(p * q) coefficients: {:?}", product.coefficients());

    // 2. Polynomial interpolation with Lagrange method
    println!("\n2. Lagrange Polynomial Interpolation");
    println!("-----------------------------------");

    // Create data points: (0,1), (1,2), (2,5)
    let x = Array::from_vec(vec![0.0, 1.0, 2.0]);
    let y = Array::from_vec(vec![1.0, 2.0, 5.0]);

    match PolynomialInterpolation::lagrange(&x, &y) {
        Ok(poly) => {
            println!(
                "Lagrange polynomial coefficients: {:?}",
                poly.coefficients()
            );

            println!("\nVerifying interpolation at data points:");
            println!("poly(0) = {}", poly.evaluate(0.0));
            println!("poly(1) = {}", poly.evaluate(1.0));
            println!("poly(2) = {}", poly.evaluate(2.0));

            println!("\nInterpolating at new points:");
            println!("poly(0.5) = {}", poly.evaluate(0.5));
            println!("poly(1.5) = {}", poly.evaluate(1.5));

            // Evaluate at multiple points
            let test_points = Array::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
            let result = poly.evaluate_array(&test_points).unwrap();
            println!("\nEvaluated at multiple points: {:?}", result.to_vec());
        }
        Err(e) => println!("Error: {}", e),
    }

    // 3. Newton polynomial interpolation
    println!("\n3. Newton Polynomial Interpolation");
    println!("----------------------------------");

    match PolynomialInterpolation::newton(&x, &y) {
        Ok(poly) => {
            println!("Newton polynomial coefficients: {:?}", poly.coefficients());

            println!("\nVerifying interpolation at data points:");
            println!("poly(0) = {}", poly.evaluate(0.0));
            println!("poly(1) = {}", poly.evaluate(1.0));
            println!("poly(2) = {}", poly.evaluate(2.0));
        }
        Err(e) => println!("Error: {}", e),
    }

    // 4. Cubic spline interpolation
    println!("\n4. Cubic Spline Interpolation");
    println!("-----------------------------");

    // Create more data points for spline
    let x = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    let y = Array::from_vec(vec![0.0, 0.5, 2.0, 1.5, 0.0]);

    match CubicSpline::new(&x, &y) {
        Ok(spline) => {
            println!("Cubic spline created successfully");

            println!("\nVerifying interpolation at knots:");
            for i in 0..x.size() {
                let xi = x.to_vec()[i];
                println!("spline({}) = {}", xi, spline.evaluate(xi).unwrap());
            }

            println!("\nInterpolating at intermediate points:");
            for i in 0..9 {
                let xi = i as f64 * 0.5;
                match spline.evaluate(xi) {
                    Ok(val) => println!("spline({}) = {}", xi, val),
                    Err(e) => println!("Error at {}: {}", xi, e),
                }
            }

            // Evaluate at multiple points
            let test_points = linspace(0.0, 4.0, 9);
            match spline.evaluate_array(&test_points) {
                Ok(result) => println!("\nEvaluated at multiple points: {:?}", result.to_vec()),
                Err(e) => println!("Error: {}", e),
            }
        }
        Err(e) => println!("Error creating spline: {}", e),
    }

    // 5. Visualize the different interpolation methods
    println!("\n5. Comparison of Interpolation Methods");
    println!("-------------------------------------");
    println!("In a graphical application, you would plot these different methods to compare:");
    println!("- Lagrange interpolation: Good for small number of points");
    println!("- Newton interpolation: More numerically stable than Lagrange");
    println!("- Cubic spline: Smoother, avoids oscillations of high-degree polynomials");
}
