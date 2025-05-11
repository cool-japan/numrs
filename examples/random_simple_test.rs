// Simple test for NumRS2 random distributions
//
// This is a minimal example to quickly verify
// the functionality of the random distributions module.

use numrs2::error::Result;
use numrs2::random::distributions::*;

fn main() -> Result<()> {
    println!("NumRS2 Random Distributions Simple Test");
    println!("======================================");

    // Set seed for reproducibility
    set_seed(42);

    // Generate and print arrays from various distributions
    println!("\nNormal(0, 1):      {:?}", normal(0.0, 1.0, &[5])?);
    println!("Beta(2, 5):        {:?}", beta(2.0, 5.0, &[5])?);
    println!("Gamma(1, 2):       {:?}", gamma(1.0, 2.0, &[5])?);
    println!("Uniform(0, 1):     {:?}", uniform(0.0, 1.0, &[5])?);
    println!("Poisson(5):        {:?}", poisson::<u32>(5.0, &[5])?);
    println!("Binomial(10, 0.5): {:?}", binomial::<u32>(10, 0.5, &[5])?);
    println!("Exponential(1):    {:?}", exponential(1.0, &[5])?);
    println!("Bernoulli(0.5):    {:?}", bernoulli(0.5, &[5])?);

    // Test multidimensional array generation
    let two_dim = normal(0.0, 1.0, &[2, 3])?;
    println!(
        "\n2D Normal array (shape {:?}):\n{:?}",
        two_dim.shape(),
        two_dim
    );

    // Verify seed reproducibility
    println!("\nChecking seed reproducibility:");
    set_seed(12345);
    let first = normal(0.0, 1.0, &[3])?;

    set_seed(12345);
    let second = normal(0.0, 1.0, &[3])?;

    println!(
        "Same seed should generate identical arrays: {}",
        first.to_vec() == second.to_vec()
    );

    println!("\nTest complete!");
    Ok(())
}
