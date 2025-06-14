#![allow(deprecated)]

use numrs2::prelude::*;

fn main() {
    println!("NumRS2 Special Functions Example");
    println!("================================\n");

    // Create a range of values to demonstrate with
    let x = linspace(-2.0, 2.0, 9);
    println!("Input values x: {:?}", x.to_vec());

    // Error Functions
    println!("\n--- Error Functions ---");
    println!("erf(x): {:?}", erf(&x).to_vec());
    println!("erfc(x): {:?}", erfc(&x).to_vec());

    // Only use values in the correct range for inverse error functions
    let y = linspace(-0.9, 0.9, 7);
    println!("Input values y (for inverse functions): {:?}", y.to_vec());
    println!("erfinv(y): {:?}", erfinv(&y).to_vec());

    // Create 1-|y| data
    let mut y_abs = y.abs().to_vec();
    for i in 0..y_abs.len() {
        y_abs[i] = 1.0 - y_abs[i];
    }
    println!(
        "erfcinv(1-|y|): {:?}",
        erfcinv(&Array::from_vec(y_abs)).to_vec()
    );

    // Gamma Functions
    println!("\n--- Gamma Functions ---");
    let z = linspace(1.0, 5.0, 5);
    println!("Input values z (positive): {:?}", z.to_vec());
    println!("gamma(z): {:?}", gamma(&z).to_vec());
    println!("gammaln(z): {:?}", gammaln(&z).to_vec());
    println!("digamma(z): {:?}", digamma(&z).to_vec());

    // Bessel Functions
    println!("\n--- Bessel Functions ---");
    let u = linspace(0.1, 5.0, 6);
    println!("Input values u (positive): {:?}", u.to_vec());
    println!("J_0(u): {:?}", bessel_j(0, &u).to_vec());
    println!("J_1(u): {:?}", bessel_j(1, &u).to_vec());
    println!("Y_0(u): {:?}", bessel_y(0, &u).to_vec());
    println!("I_0(u): {:?}", bessel_i(0, &u).to_vec());
    println!("K_0(u): {:?}", bessel_k(0, &u).to_vec());

    // Elliptic Integrals
    println!("\n--- Elliptic Integrals ---");
    let m = linspace(0.0, 0.9, 5);
    println!("Input values m: {:?}", m.to_vec());
    println!("K(m): {:?}", ellipk(&m).to_vec());
    println!("E(m): {:?}", ellipe(&m).to_vec());

    // Example application: Normal distribution CDF
    println!("\n--- Application: Normal Distribution CDF ---");
    let w = linspace(-3.0, 3.0, 7);
    println!("For standard normal distribution N(0,1)");
    println!("x values: {:?}", w.to_vec());

    // Calculate CDF using error function: CDF(x) = 0.5 * (1 + erf(x/sqrt(2)))
    let sqrt2 = (2.0_f64).sqrt();

    // Create x/sqrt(2) manually
    let mut scaled_w = w.to_vec();
    for i in 0..scaled_w.len() {
        scaled_w[i] = scaled_w[i] / sqrt2;
    }

    // Calculate normal CDF
    let erf_values = erf(&Array::from_vec(scaled_w)).to_vec();
    let mut normal_cdf = Vec::with_capacity(erf_values.len());
    for val in erf_values {
        normal_cdf.push(0.5 * (1.0 + val));
    }

    println!("CDF values: {:?}", normal_cdf);
}
