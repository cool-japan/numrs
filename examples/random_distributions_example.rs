// Example demonstrating various random distributions in NumRS2
//
// This example shows how to use the random distributions module in NumRS2,
// including setting seeds for reproducibility and generating arrays from
// different probability distributions. Advanced distributions leverage
// SciRS2 integration when available.

use numrs2::array::Array;
use numrs2::error::Result;
use numrs2::random::distributions::*;

// Import the SciRS2 integration for advanced distributions (removed wildcard to avoid conflicts)
use std::time::Instant;

fn main() -> Result<()> {
    println!("NumRS2 Random Distributions Example");
    println!("====================================");

    // Set a seed for reproducibility
    println!("\nSetting a seed for reproducibility...");
    set_seed(12345);

    // Part 1: Continuous distributions
    println!("\nPart 1: Continuous Distributions");
    println!("--------------------------------");

    // Normal (Gaussian) distribution
    let normal_samples = normal(0.0, 1.0, &[5])?;
    println!("Normal(0, 1) samples: {:?}", normal_samples);

    // Standard normal distribution (mean=0, std=1)
    let std_normal_samples = standard_normal::<f64>(&[5])?;
    println!("Standard Normal samples: {:?}", std_normal_samples);

    // Beta distribution
    let beta_samples = beta(2.0, 5.0, &[5])?;
    println!("Beta(2, 5) samples: {:?}", beta_samples);

    // Gamma distribution
    let gamma_samples = gamma(1.0, 2.0, &[5])?;
    println!("Gamma(1, 2) samples: {:?}", gamma_samples);

    // Chi-square distribution
    let chisquare_samples = chisquare(2.0, &[5])?;
    println!("Chi-square(2) samples: {:?}", chisquare_samples);

    // Exponential distribution
    let exponential_samples = exponential(1.0, &[5])?;
    println!("Exponential(1) samples: {:?}", exponential_samples);

    // Uniform distribution
    let uniform_samples = uniform(0.0, 1.0, &[5])?;
    println!("Uniform(0, 1) samples: {:?}", uniform_samples);

    // Log-normal distribution
    let lognormal_samples = lognormal(0.0, 1.0, &[5])?;
    println!("LogNormal(0, 1) samples: {:?}", lognormal_samples);

    // Cauchy distribution
    let cauchy_samples = cauchy(0.0, 1.0, &[5])?;
    println!("Cauchy(0, 1) samples: {:?}", cauchy_samples);

    // Student's t-distribution
    let student_t_samples = student_t(5.0, &[5])?;
    println!("Student's t(5) samples: {:?}", student_t_samples);

    // Weibull distribution
    let weibull_samples = weibull(1.0, 1.0, &[5])?;
    println!("Weibull(1, 1) samples: {:?}", weibull_samples);

    // Pareto distribution
    let pareto_samples = pareto(2.0, &[5])?;
    println!("Pareto(2) samples: {:?}", pareto_samples);

    // Laplace distribution
    let laplace_samples = laplace(0.0, 1.0, &[5])?;
    println!("Laplace(0, 1) samples: {:?}", laplace_samples);

    // Gumbel distribution
    let gumbel_samples = gumbel(0.0, 1.0, &[5])?;
    println!("Gumbel(0, 1) samples: {:?}", gumbel_samples);

    // Logistic distribution
    let logistic_samples = logistic(0.0, 1.0, &[5])?;
    println!("Logistic(0, 1) samples: {:?}", logistic_samples);

    // Rayleigh distribution
    let rayleigh_samples = rayleigh(1.0, &[5])?;
    println!("Rayleigh(1) samples: {:?}", rayleigh_samples);

    // Triangular distribution with error handling
    println!("\nAttempting to create Triangular(0, 2, 10) distribution...");
    match triangular(0.0, 2.0, 10.0, &[5]) {
        Ok(samples) => println!("Triangular(0, 2, 10) samples: {:?}", samples),
        Err(e) => println!("Error creating triangular distribution: {:?}", e),
    }

    // PERT distribution
    println!("\nAttempting to create PERT(0, 5, 10) distribution...");
    match pert(0.0, 5.0, 10.0, &[5]) {
        Ok(samples) => println!("PERT(0, 5, 10) samples: {:?}", samples),
        Err(e) => println!("Error creating PERT distribution: {:?}", e),
    }

    // Wald (inverse Gaussian) distribution
    let wald_samples = wald(1.0, 1.0, &[5])?;
    println!("Wald(1, 1) samples: {:?}", wald_samples);

    // Part 2: Discrete distributions
    println!("\nPart 2: Discrete Distributions");
    println!("-----------------------------");

    // Binomial distribution
    let binomial_samples = binomial::<u32>(10, 0.5, &[5])?;
    println!("Binomial(10, 0.5) samples: {:?}", binomial_samples);

    // Poisson distribution
    let poisson_samples = poisson::<u32>(5.0, &[5])?;
    println!("Poisson(5) samples: {:?}", poisson_samples);

    // Bernoulli distribution
    let bernoulli_samples = bernoulli(0.5, &[5])?;
    println!("Bernoulli(0.5) samples: {:?}", bernoulli_samples);

    // Negative binomial distribution
    let neg_binomial_samples = negative_binomial::<u32>(5.0, 0.5, &[5])?;
    println!(
        "Negative Binomial(5, 0.5) samples: {:?}",
        neg_binomial_samples
    );

    // Geometric distribution
    let geometric_samples = geometric::<u32>(0.5, &[5])?;
    println!("Geometric(0.5) samples: {:?}", geometric_samples);

    // Zipf distribution
    let zipf_samples = zipf::<u32>(2.0, &[5])?;
    println!("Zipf(2) samples: {:?}", zipf_samples);

    // Logseries distribution
    let logseries_samples = logseries::<u32>(0.5, &[5])?;
    println!("LogSeries(0.5) samples: {:?}", logseries_samples);

    // Hypergeometric distribution
    println!("\nAttempting to create Hypergeometric(20, 30, 10) distribution...");
    match hypergeometric::<u32>(20, 30, 10, &[5]) {
        Ok(samples) => println!("Hypergeometric(20, 30, 10) samples: {:?}", samples),
        Err(e) => println!("Error creating hypergeometric distribution: {:?}", e),
    }

    // Part 3: Multivariate distributions
    println!("\nPart 3: Multivariate Distributions");
    println!("--------------------------------");

    // Dirichlet distribution
    let alpha = vec![1.0, 1.0, 1.0];
    let dirichlet_samples = dirichlet(&alpha, &[2])?;
    println!("Dirichlet([1, 1, 1]) samples:\n{:?}", dirichlet_samples);
    println!(
        "Shape of Dirichlet samples: {:?}",
        dirichlet_samples.shape()
    );

    // Multivariate normal distribution
    let mean = vec![0.0, 0.0];
    let cov_data = vec![1.0, 0.5, 0.5, 1.0];
    let cov = Array::from_vec(cov_data).reshape(&[2, 2]);
    let mvn_samples = multivariate_normal(&mean, &cov, Some(&[3]))?;
    println!("Multivariate Normal samples:\n{:?}", mvn_samples);
    println!(
        "Shape of Multivariate Normal samples: {:?}",
        mvn_samples.shape()
    );

    // Multinomial distribution
    let pvals = vec![0.2, 0.3, 0.5];
    let multinomial_samples = multinomial::<u32>(10, &pvals, None)?;
    println!(
        "Multinomial(10, [0.2, 0.3, 0.5]) samples: {:?}",
        multinomial_samples
    );
    println!(
        "Shape of Multinomial samples: {:?}",
        multinomial_samples.shape()
    );

    let multinomial_multiple = multinomial::<u32>(10, &pvals, Some(&[2]))?;
    println!("Multiple Multinomial samples:\n{:?}", multinomial_multiple);
    println!(
        "Shape of multiple Multinomial samples: {:?}",
        multinomial_multiple.shape()
    );

    // Part 4: Distribution Properties
    println!("\nPart 4: Demonstration of Distribution Properties");
    println!("--------------------------------------------");

    // Let's verify the mean and variance of normal distribution
    println!("\nVerifying properties of Normal(0, 1) distribution with 10,000 samples:");
    let start_time = Instant::now();
    let large_normal = normal(0.0, 1.0, &[10000])?;
    let elapsed = start_time.elapsed();

    let data = large_normal.to_vec();
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;

    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    println!("Generation time: {:?}", elapsed);
    println!("Sample mean: {:.6} (expected: 0)", mean);
    println!("Sample standard deviation: {:.6} (expected: 1)", std_dev);

    // Part 5: Seed effects on reproducibility
    println!("\nPart 5: Seed Effects on Reproducibility");
    println!("------------------------------------");

    // Try different seeds
    set_seed(42);
    let samples1 = normal(0.0, 1.0, &[5])?;

    set_seed(42);
    let samples2 = normal(0.0, 1.0, &[5])?;

    set_seed(84);
    let samples3 = normal(0.0, 1.0, &[5])?;

    println!("Samples with seed=42 (first run): {:?}", samples1);
    println!("Samples with seed=42 (second run): {:?}", samples2);
    println!("Samples with seed=84: {:?}", samples3);
    println!(
        "Are the two runs with seed=42 identical? {}",
        samples1.to_vec() == samples2.to_vec()
    );
    println!(
        "Are runs with different seeds different? {}",
        samples1.to_vec() != samples3.to_vec()
    );

    // Part 6: Shape effects
    println!("\nPart 6: Generating Multi-dimensional Arrays");
    println!("----------------------------------------");

    // Generate 2D normal array
    let normal_2d = normal(0.0, 1.0, &[3, 4])?;
    println!("2D Normal array shape: {:?}", normal_2d.shape());
    println!("2D Normal array:\n{:?}", normal_2d);

    // Generate 3D uniform array
    let uniform_3d = uniform(0.0, 1.0, &[2, 2, 2])?;
    println!("3D Uniform array shape: {:?}", uniform_3d.shape());
    println!("3D Uniform array:\n{:?}", uniform_3d);

    // Part 7: SciRS2 Integration for Advanced Distributions
    println!("\nPart 7: SciRS2 Integration for Advanced Distributions");
    println!("----------------------------------------------");

    // Demonstrate the advanced distributions from SciRS2 integration
    println!("\nAdvanced distributions from SciRS2 integration:");

    // Set a seed for reproducibility
    set_seed(12345);

    // Noncentral chi-square distribution
    println!("\nAttempting to create Noncentral Chi-square(2, 1) distribution...");
    match noncentral_chisquare(2.0, 1.0, &[5]) {
        Ok(samples) => {
            println!("Non-central Chi-square(2, 1) samples: {:?}", samples);
            println!("Shape of samples: {:?}", samples.shape());

            // In a non-central chi-square with df=2 and nonc=1, the mean should be close to 3
            let data = samples.to_vec();
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            println!("Sample mean: {:.4} (expected: approx 3)", mean);
        }
        Err(e) => println!("Note on noncentral chi-square: {}", e),
    }

    // Noncentral F distribution
    println!("\nAttempting to create Noncentral F(2, 5, 1) distribution...");
    match noncentral_f(2.0, 5.0, 1.0, &[5]) {
        Ok(samples) => {
            println!("Non-central F(2, 5, 1) samples: {:?}", samples);
            println!("Shape of samples: {:?}", samples.shape());

            // Verify all values are positive
            let min_val = samples
                .to_vec()
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            println!("Minimum value: {:.4} (should be > 0)", min_val);
        }
        Err(e) => println!("Note on noncentral F: {}", e),
    }

    // Von Mises distribution
    println!("\nAttempting to create von Mises(0, 1) distribution...");
    match vonmises(0.0, 1.0, &[5]) {
        Ok(samples) => {
            println!("von Mises(0, 1) samples: {:?}", samples);
            println!("Shape of samples: {:?}", samples.shape());

            // Von Mises samples should be in the range [-PI, PI]
            let data = samples.to_vec();
            let in_range = data
                .iter()
                .all(|&x| x >= -std::f64::consts::PI && x <= std::f64::consts::PI);
            println!("All values in range [-PI, PI]: {}", in_range);
        }
        Err(e) => println!("Note on von Mises: {}", e),
    }

    // Maxwell distribution
    println!("\nAttempting to create Maxwell(1) distribution...");
    match maxwell(1.0, &[5]) {
        Ok(samples) => {
            println!("Maxwell(1) samples: {:?}", samples);
            println!("Shape of samples: {:?}", samples.shape());

            // Maxwell samples should be positive
            let data = samples.to_vec();
            let all_positive = data.iter().all(|&x| x > 0.0);
            println!("All values positive: {}", all_positive);
        }
        Err(e) => println!("Note on Maxwell: {}", e),
    }

    // Truncated normal distribution
    println!("\nAttempting to create Truncated Normal(0, 1, -2, 2) distribution...");
    match truncated_normal(0.0, 1.0, -2.0, 2.0, &[5]) {
        Ok(samples) => {
            println!("Truncated Normal(0, 1, -2, 2) samples: {:?}", samples);
            println!("Shape of samples: {:?}", samples.shape());

            // All values should be within the truncation bounds
            let data = samples.to_vec();
            let in_bounds = data.iter().all(|&x| x >= -2.0 && x <= 2.0);
            println!("All values within bounds [-2, 2]: {}", in_bounds);
        }
        Err(e) => println!("Note on truncated normal: {}", e),
    }

    // Multivariate normal with rotation
    println!("\nAttempting to create multivariate normal with rotation...");
    let mean = vec![0.0, 0.0];
    let cov_data = vec![1.0, 0.5, 0.5, 1.0];
    let cov = Array::from_vec(cov_data).reshape(&[2, 2]);

    // Create a rotation matrix for 45 degrees
    let rotation_data = vec![
        0.7071, 0.7071, // cos(45°), sin(45°)
        -0.7071, 0.7071, // -sin(45°), cos(45°)
    ];
    let rotation = Array::from_vec(rotation_data).reshape(&[2, 2]);

    match multivariate_normal_with_rotation(&mean, &cov, Some(&[3]), Some(&rotation)) {
        Ok(samples) => {
            println!("Multivariate normal with rotation samples:\n{:?}", samples);
            println!("Shape of samples: {:?}", samples.shape());
        }
        Err(e) => println!("Note on multivariate normal with rotation: {}", e),
    }

    // Generate a larger sample to demonstrate distribution properties
    println!("\nGenerating 1000 samples from von Mises(0, 5) to demonstrate concentration:");
    match vonmises(0.0, 5.0, &[1000]) {
        Ok(samples) => {
            // Calculate how concentrated the samples are around the mean
            let data = samples.to_vec();
            let count_within_pi_6 = data
                .iter()
                .filter(|&&x: &&f64| x.abs() <= std::f64::consts::PI / 6.0)
                .count();
            let percentage = 100.0 * (count_within_pi_6 as f64) / 1000.0;

            println!(
                "Percentage of samples within [-π/6, π/6]: {:.1}%",
                percentage
            );
            println!("With kappa=5, we expect a higher concentration around the mean direction.");
        }
        Err(e) => println!("Note on von Mises: {}", e),
    }

    // Instructions for enabling SciRS2 integration
    println!("\nAdvanced distributions note:");
    println!("To enable SciRS2 integration for these advanced distributions:");
    println!("1. Add the scirs2-stats and scirs2-core dependencies to your project");
    println!("2. Enable the 'scirs' feature when building NumRS2");
    println!("   cargo build --features scirs");
    println!("3. Ensure the SciRS2 library is installed in your environment");

    println!("\nExample complete. NumRS2 random distributions successfully demonstrated!");

    Ok(())
}
