#[cfg(test)]
mod tests {
    use numrs2::array::Array;
    use numrs2::random::{self, set_seed};

    /// This is a helper test to generate the reference values for all the distributions
    /// Run this test with:
    /// cargo test --test test_random_values_collector -- --nocapture
    #[test]
    fn collect_random_reference_values() {
        // Set the seed to 42 for reproducibility
        set_seed(42);

        // Normal distribution
        println!("// Normal distribution reference values");
        let normal_samples = random::normal(0.0, 1.0, &[5]).unwrap();
        println!("let expected_values = vec![");
        for val in normal_samples.to_vec() {
            println!("    {:.16},", val);
        }
        println!("];\n");

        // Beta distribution
        println!("// Beta distribution reference values");
        let beta_samples = random::beta(2.0, 5.0, &[5]).unwrap();
        println!("let expected_values = vec![");
        for val in beta_samples.to_vec() {
            println!("    {:.16},", val);
        }
        println!("];\n");

        // Uniform distribution
        println!("// Uniform distribution reference values");
        let uniform_samples = random::uniform(0.0, 1.0, &[5]).unwrap();
        println!("let expected_values = vec![");
        for val in uniform_samples.to_vec() {
            println!("    {:.16},", val);
        }
        println!("];\n");

        // Gamma distribution
        println!("// Gamma distribution reference values");
        let gamma_samples = random::gamma(2.0, 3.0, &[5]).unwrap();
        println!("let expected_values = vec![");
        for val in gamma_samples.to_vec() {
            println!("    {:.16},", val);
        }
        println!("];\n");

        // Integers distribution
        println!("// Integers distribution reference values");
        let int_samples = random::integers(1, 100, &[10]).unwrap();
        println!("let expected_values = vec![");
        for val in int_samples.to_vec() {
            println!("    {},", val);
        }
        println!("];\n");

        // Binomial distribution
        println!("// Binomial distribution reference values");
        let binomial_samples = random::binomial::<u64>(20, 0.3, &[5]).unwrap();
        println!("let expected_values = vec![");
        for val in binomial_samples.to_vec() {
            println!("    {},", val);
        }
        println!("];\n");

        // Poisson distribution
        println!("// Poisson distribution reference values");
        let poisson_samples = random::poisson::<u64>(5.0, &[5]).unwrap();
        println!("let expected_values = vec![");
        for val in poisson_samples.to_vec() {
            println!("    {},", val);
        }
        println!("];\n");

        // Multivariate normal distribution
        println!("// Multivariate normal distribution reference values");
        let mean = vec![1.0, 2.0];
        let cov_data = vec![1.0, 0.5, 0.5, 2.0];
        let cov = Array::from_vec(cov_data).reshape(&[2, 2]);
        let mvn_samples = random::multivariate_normal(&mean, &cov, Some(&[2])).unwrap();
        println!("let expected_values = vec![");
        for val in mvn_samples.to_vec() {
            println!("    {:.16},", val);
        }
        println!("];\n");

        // Exponential distribution
        println!("// Exponential distribution reference values");
        let exp_samples = random::exponential(2.0, &[5]).unwrap();
        println!("let expected_values = vec![");
        for val in exp_samples.to_vec() {
            println!("    {:.16},", val);
        }
        println!("];\n");

        // Lognormal distribution
        println!("// Lognormal distribution reference values");
        let lognormal_samples = random::lognormal(0.0, 1.0, &[5]).unwrap();
        println!("let expected_values = vec![");
        for val in lognormal_samples.to_vec() {
            println!("    {:.16},", val);
        }
        println!("];\n");

        // Weibull distribution
        println!("// Weibull distribution reference values");
        let weibull_samples = random::weibull(2.0, 3.0, &[5]).unwrap();
        println!("let expected_values = vec![");
        for val in weibull_samples.to_vec() {
            println!("    {:.16},", val);
        }
        println!("];");
    }
}
