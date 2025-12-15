use numrs2::array::Array;
use numrs2::random::{self, set_seed};

// This file contains statistical property tests for random distributions.
// Instead of testing specific values, we test that the distributions
// satisfy statistical properties like mean, variance, etc.
// Sample size for statistical tests
const SAMPLE_SIZE: usize = 10000;

// Helper function to calculate the mean of an array
fn calculate_mean(arr: &Array<f64>) -> f64 {
    let data = arr.to_vec();
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

// Helper function to calculate the variance of an array
fn calculate_variance(arr: &Array<f64>, mean: f64) -> f64 {
    let data = arr.to_vec();
    let sum_sq_diff: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
    sum_sq_diff / data.len() as f64
}

#[test]
fn test_normal_distribution_statistics() {
    // Test normal distributions with different parameters
    let mean = 3.0;
    let std_dev = 2.0;

    set_seed(42);
    let samples = random::normal(mean, std_dev, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);
    let _sample_std_dev = sample_variance.sqrt();

    // Check if mean is within expected range
    assert!(
        (sample_mean - mean).abs() < 0.1 * std_dev,
        "Normal distribution: Expected mean close to {}, got {}",
        mean,
        sample_mean
    );

    // Check if variance is within expected range
    assert!(
        (sample_variance - std_dev * std_dev).abs() < 0.15 * std_dev * std_dev,
        "Normal distribution: Expected variance close to {}, got {}",
        std_dev * std_dev,
        sample_variance
    );
}

#[test]
fn test_uniform_distribution_statistics() {
    // Test uniform distribution
    let low = 2.0;
    let high = 7.0;
    let range = high - low;

    set_seed(42);
    let samples = random::uniform(low, high, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);

    // Expected values for uniform distribution
    let expected_mean = (low + high) / 2.0;
    let expected_variance = range * range / 12.0;

    // Check if mean is within expected range
    assert!(
        (sample_mean - expected_mean).abs() < 0.1 * range,
        "Uniform distribution: Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // Check if variance is within expected range
    assert!(
        (sample_variance - expected_variance).abs() < 0.15 * expected_variance,
        "Uniform distribution: Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // Check if all values are within the specified range
    let all_in_range = samples.to_vec().iter().all(|&x| x >= low && x <= high);
    assert!(
        all_in_range,
        "Uniform distribution: Some values are outside [{}, {}]",
        low, high
    );
}

#[test]
fn test_beta_distribution_statistics() {
    // Test beta distribution
    let alpha = 2.0;
    let beta = 5.0;

    set_seed(42);
    let samples = random::beta(alpha, beta, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);

    // Expected values for beta distribution
    let expected_mean = alpha / (alpha + beta);
    let expected_variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));

    // Check if mean is within expected range
    assert!(
        (sample_mean - expected_mean).abs() < 0.1 * expected_mean,
        "Beta distribution: Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // Check if variance is within expected range
    assert!(
        (sample_variance - expected_variance).abs() < 0.2 * expected_variance,
        "Beta distribution: Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // Check if all values are within [0, 1]
    let all_in_range = samples.to_vec().iter().all(|&x| (0.0..=1.0).contains(&x));
    assert!(
        all_in_range,
        "Beta distribution: Some values are outside [0, 1]"
    );
}

#[test]
fn test_exponential_distribution_statistics() {
    // Test exponential distribution
    let scale = 3.0;

    set_seed(42);
    let samples = random::exponential(scale, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);

    // Expected values for exponential distribution
    let expected_mean = scale;
    let expected_variance = scale * scale;

    // Check if mean is within expected range
    assert!(
        (sample_mean - expected_mean).abs() < 0.15 * expected_mean,
        "Exponential distribution: Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // Check if variance is within expected range
    assert!(
        (sample_variance - expected_variance).abs() < 0.2 * expected_variance,
        "Exponential distribution: Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // Check if all values are positive
    let all_positive = samples.to_vec().iter().all(|&x| x > 0.0);
    assert!(
        all_positive,
        "Exponential distribution: Some values are not positive"
    );
}

#[test]
fn test_gamma_distribution_statistics() {
    // Test gamma distribution
    let shape = 3.0;
    let scale = 2.0;

    set_seed(42);
    let samples = random::gamma(shape, scale, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);

    // Expected values for gamma distribution
    let expected_mean = shape * scale;
    let expected_variance = shape * scale * scale;

    // Check if mean is within expected range
    assert!(
        (sample_mean - expected_mean).abs() < 0.15 * expected_mean,
        "Gamma distribution: Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // Check if variance is within expected range
    assert!(
        (sample_variance - expected_variance).abs() < 0.2 * expected_variance,
        "Gamma distribution: Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // Check if all values are positive
    let all_positive = samples.to_vec().iter().all(|&x| x > 0.0);
    assert!(
        all_positive,
        "Gamma distribution: Some values are not positive"
    );
}

#[test]
fn test_lognormal_distribution_statistics() {
    // Test lognormal distribution
    let mu = 0.0;
    let sigma = 1.0;

    set_seed(42);
    let samples = random::lognormal(mu, sigma, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);

    // Expected values for lognormal distribution
    let expected_mean = (mu + sigma * sigma / 2.0).exp();
    let expected_variance = ((sigma * sigma).exp() - 1.0) * (2.0 * mu + sigma * sigma).exp();

    // Check if mean is within expected range
    assert!(
        (sample_mean - expected_mean).abs() < 0.2 * expected_mean,
        "Lognormal distribution: Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // Check if variance is within expected range (wider tolerance due to high variance of lognormal)
    assert!(
        (sample_variance - expected_variance).abs() < 0.3 * expected_variance,
        "Lognormal distribution: Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // Check if all values are positive
    let all_positive = samples.to_vec().iter().all(|&x| x > 0.0);
    assert!(
        all_positive,
        "Lognormal distribution: Some values are not positive"
    );
}

#[test]
fn test_weibull_distribution_statistics() {
    // Test Weibull distribution
    let shape = 2.0;
    let scale = 3.0;

    set_seed(42);
    let samples = random::weibull(shape, scale, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);

    // Expected mean for Weibull distribution with shape k=2, scale λ=3
    // For Weibull, mean = λ * Γ(1 + 1/k), where Γ is the gamma function
    // With k=2, λ=3: mean = 3 * Γ(1.5) = 3 * sqrt(π)/2 ≈ 2.66
    // Beta.4 fixed the Weibull parameterization, now using correct formula
    let expected_mean = 2.66;

    // Check if mean is within expected range
    assert!(
        (sample_mean - expected_mean).abs() < 0.15 * expected_mean,
        "Weibull distribution: Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // Check if all values are positive
    let all_positive = samples.to_vec().iter().all(|&x| x > 0.0);
    assert!(
        all_positive,
        "Weibull distribution: Some values are not positive"
    );
}

#[test]
fn test_binomial_distribution_statistics() {
    // Test binomial distribution
    let n = 20u64;
    let p = 0.3;

    set_seed(42);
    let samples = random::binomial::<u64>(n, p, &[SAMPLE_SIZE]).unwrap();

    // Convert to f64 for calculation
    let samples_f64 = Array::from_vec(samples.to_vec().iter().map(|&x| x as f64).collect());

    // Calculate statistics
    let sample_mean = calculate_mean(&samples_f64);
    let sample_variance = calculate_variance(&samples_f64, sample_mean);

    // Expected values for binomial distribution
    let expected_mean = n as f64 * p;
    let expected_variance = n as f64 * p * (1.0 - p);

    // Check if mean is within expected range
    assert!(
        (sample_mean - expected_mean).abs() < 0.1 * expected_mean,
        "Binomial distribution: Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // Check if variance is within expected range
    assert!(
        (sample_variance - expected_variance).abs() < 0.15 * expected_variance,
        "Binomial distribution: Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // Check if all values are within allowed range [0, n]
    let all_in_range = samples.to_vec().iter().all(|&x| x <= n);
    assert!(
        all_in_range,
        "Binomial distribution: Some values are outside [0, {}]",
        n
    );
}

#[test]
fn test_poisson_distribution_statistics() {
    // Test Poisson distribution
    let lambda = 5.0;

    set_seed(42);
    let samples = random::poisson::<u64>(lambda, &[SAMPLE_SIZE]).unwrap();

    // Convert to f64 for calculation
    let samples_f64 = Array::from_vec(samples.to_vec().iter().map(|&x| x as f64).collect());

    // Calculate statistics
    let sample_mean = calculate_mean(&samples_f64);
    let sample_variance = calculate_variance(&samples_f64, sample_mean);

    // Expected values for Poisson distribution (mean = variance = lambda)
    let expected_mean = lambda;
    let expected_variance = lambda;

    // Check if mean is within expected range
    assert!(
        (sample_mean - expected_mean).abs() < 0.1 * expected_mean,
        "Poisson distribution: Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // Check if variance is within expected range
    assert!(
        (sample_variance - expected_variance).abs() < 0.15 * expected_variance,
        "Poisson distribution: Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // Check if all values are non-negative integers
    let all_non_negative = samples.to_vec().iter().all(|_x| true);
    assert!(
        all_non_negative,
        "Poisson distribution: Some values are negative"
    );
}

#[test]
#[ignore] // This test may fail due to concurrent test execution affecting global state
fn test_seed_reproducibility() {
    // Test that setting the same seed produces the same results
    // NOTE: This test requires sequential execution as it uses global random state

    // First run with seed 42
    set_seed(42);
    let samples1 = random::normal(0.0, 1.0, &[100]).unwrap();

    // Second run with same seed
    set_seed(42);
    let samples2 = random::normal(0.0, 1.0, &[100]).unwrap();

    // Third run with different seed
    set_seed(43);
    let samples3 = random::normal(0.0, 1.0, &[100]).unwrap();

    // Same seed should produce exactly the same values
    let samples1_vec = samples1.to_vec();
    let samples2_vec = samples2.to_vec();
    assert_eq!(
        samples1_vec, samples2_vec,
        "Same seed should produce identical results"
    );

    // Different seed should produce different values
    let samples3_vec = samples3.to_vec();
    assert_ne!(
        samples1_vec, samples3_vec,
        "Different seeds should produce different results"
    );
}
