use numrs2::array::Array;
use numrs2::random::{self, set_seed};

/// This is a helper file to generate reference values from the current implementation
/// to be used in the test_random_reference.rs file.

fn main() {
    // Set the seed to 42 for reproducibility
    set_seed(42);

    // Generate values for normal distribution
    println!("Normal distribution reference values:");
    let normal_samples = random::normal(0.0, 1.0, &[5]).unwrap();
    let values = normal_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {:.16},", val);
    }
    println!("];");
    println!();

    // Generate values for beta distribution
    println!("Beta distribution reference values:");
    let beta_samples = random::beta(2.0, 5.0, &[5]).unwrap();
    let values = beta_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {:.16},", val);
    }
    println!("];");
    println!();

    // Generate values for uniform distribution
    println!("Uniform distribution reference values:");
    let uniform_samples = random::uniform(0.0, 1.0, &[5]).unwrap();
    let values = uniform_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {:.16},", val);
    }
    println!("];");
    println!();

    // Generate values for gamma distribution
    println!("Gamma distribution reference values:");
    let gamma_samples = random::gamma(2.0, 3.0, &[5]).unwrap();
    let values = gamma_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {:.16},", val);
    }
    println!("];");
    println!();

    // Generate values for integers distribution
    println!("Integers distribution reference values:");
    let int_samples = random::integers(1, 100, &[10]).unwrap();
    let values = int_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {},", val);
    }
    println!("];");
    println!();

    // Generate values for binomial distribution
    println!("Binomial distribution reference values:");
    let binomial_samples = random::binomial::<u64>(20, 0.3, &[5]).unwrap();
    let values = binomial_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {},", val);
    }
    println!("];");
    println!();

    // Generate values for Poisson distribution
    println!("Poisson distribution reference values:");
    let poisson_samples = random::poisson::<u64>(5.0, &[5]).unwrap();
    let values = poisson_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {},", val);
    }
    println!("];");
    println!();

    // Generate values for multivariate normal distribution
    println!("Multivariate normal distribution reference values:");
    let mean = vec![1.0, 2.0];
    let cov_data = vec![1.0, 0.5, 0.5, 2.0];
    let cov = Array::from_vec(cov_data).reshape(&[2, 2]);
    let mvn_samples = random::multivariate_normal(&mean, &cov, Some(&[2])).unwrap();
    let values = mvn_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {:.16},", val);
    }
    println!("];");
    println!();

    // Generate values for exponential distribution
    println!("Exponential distribution reference values:");
    let exp_samples = random::exponential(2.0, &[5]).unwrap();
    let values = exp_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {:.16},", val);
    }
    println!("];");
    println!();

    // Generate values for lognormal distribution
    println!("Lognormal distribution reference values:");
    let lognormal_samples = random::lognormal(0.0, 1.0, &[5]).unwrap();
    let values = lognormal_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {:.16},", val);
    }
    println!("];");
    println!();

    // Generate values for Weibull distribution
    println!("Weibull distribution reference values:");
    let weibull_samples = random::weibull(2.0, 3.0, &[5]).unwrap();
    let values = weibull_samples.to_vec();
    println!("let expected_values = vec![");
    for val in values {
        println!("    {:.16},", val);
    }
    println!("];");
}
