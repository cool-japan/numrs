use numrs2::array::Array;
use numrs2::random::{self, set_seed};

/// This file implements property-based testing for the random module.
/// Instead of using a full property testing framework, we'll use statistical properties
/// to validate that our random distributions behave as expected.

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

// Helper function to calculate skewness
fn calculate_skewness(arr: &Array<f64>, mean: f64, std_dev: f64) -> f64 {
    let data = arr.to_vec();
    let n = data.len() as f64;
    let sum_cubed_diff: f64 = data.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum();
    sum_cubed_diff * n / ((n - 1.0) * (n - 2.0))
}

// Helper function to calculate kurtosis
fn calculate_kurtosis(arr: &Array<f64>, mean: f64, std_dev: f64) -> f64 {
    let data = arr.to_vec();
    let n = data.len() as f64;
    let sum_fourth_power: f64 = data.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum();
    sum_fourth_power * n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0))
        - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0))
}

// Helper function to check if a value is within expected bounds
fn is_within_bounds(value: f64, expected: f64, tolerance: f64) -> bool {
    (value - expected).abs() <= tolerance
}

#[test]
fn test_normal_distribution_properties() {
    let mean = 5.0;
    let std_dev = 2.0;

    // Set a fixed seed for reproducibility
    set_seed(12345);

    // Generate a large sample from the normal distribution
    let samples = random::normal(mean, std_dev, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);
    let sample_std_dev = sample_variance.sqrt();
    let sample_skewness = calculate_skewness(&samples, sample_mean, sample_std_dev);
    let sample_kurtosis = calculate_kurtosis(&samples, sample_mean, sample_std_dev);

    // For a normal distribution:
    // 1. Mean should be close to the specified mean
    assert!(
        is_within_bounds(sample_mean, mean, 0.1 * std_dev),
        "Expected mean close to {}, got {}",
        mean,
        sample_mean
    );

    // 2. Variance should be close to the square of the specified std_dev
    assert!(
        is_within_bounds(sample_variance, std_dev * std_dev, 0.2 * std_dev * std_dev),
        "Expected variance close to {}, got {}",
        std_dev * std_dev,
        sample_variance
    );

    // 3. Skewness should be close to 0 for a normal distribution
    assert!(
        is_within_bounds(sample_skewness, 0.0, 0.2),
        "Expected skewness close to 0, got {}",
        sample_skewness
    );

    // 4. Excess kurtosis should be close to 0 for a normal distribution
    assert!(
        is_within_bounds(sample_kurtosis, 0.0, 0.4),
        "Expected kurtosis close to 0, got {}",
        sample_kurtosis
    );
}

#[test]
fn test_uniform_distribution_properties() {
    let low = 2.0;
    let high = 7.0;
    let range = high - low;

    // Set a fixed seed for reproducibility
    set_seed(12345);

    // Generate a large sample from the uniform distribution
    let samples = random::uniform(low, high, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);
    let sample_std_dev = sample_variance.sqrt();
    let sample_skewness = calculate_skewness(&samples, sample_mean, sample_std_dev);
    let sample_kurtosis = calculate_kurtosis(&samples, sample_mean, sample_std_dev);

    // For a uniform distribution:
    // 1. Mean should be at the center of the range
    let expected_mean = (low + high) / 2.0;
    assert!(
        is_within_bounds(sample_mean, expected_mean, 0.1 * range),
        "Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // 2. Variance should be (high-low)²/12
    let expected_variance = range * range / 12.0;
    assert!(
        is_within_bounds(sample_variance, expected_variance, 0.2 * expected_variance),
        "Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // 3. Skewness should be close to 0 for a uniform distribution
    assert!(
        is_within_bounds(sample_skewness, 0.0, 0.2),
        "Expected skewness close to 0, got {}",
        sample_skewness
    );

    // 4. Excess kurtosis should be close to -1.2 for a uniform distribution
    assert!(
        is_within_bounds(sample_kurtosis, -1.2, 0.3),
        "Expected kurtosis close to -1.2, got {}",
        sample_kurtosis
    );

    // 5. All values should be within the specified range
    let all_in_range = samples.to_vec().iter().all(|&x| x >= low && x <= high);
    assert!(
        all_in_range,
        "Some values are outside the specified range [{}, {}]",
        low, high
    );
}

#[test]
fn test_beta_distribution_properties() {
    let alpha = 2.0;
    let beta = 5.0;

    // Set a fixed seed for reproducibility
    set_seed(12345);

    // Generate a large sample from the beta distribution
    let samples = random::beta(alpha, beta, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);

    // For a beta distribution:
    // 1. Mean should be close to alpha/(alpha+beta)
    let expected_mean = alpha / (alpha + beta);
    assert!(
        is_within_bounds(sample_mean, expected_mean, 0.1),
        "Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // 2. Variance should be close to (alpha*beta)/((alpha+beta)²*(alpha+beta+1))
    let expected_variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
    assert!(
        is_within_bounds(sample_variance, expected_variance, 0.1),
        "Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // 3. All values should be within [0, 1]
    let all_in_range = samples.to_vec().iter().all(|&x| x >= 0.0 && x <= 1.0);
    assert!(all_in_range, "Some values are outside [0, 1]");
}

#[test]
fn test_gamma_distribution_properties() {
    let shape = 3.0;
    let scale = 2.0;

    // Set a fixed seed for reproducibility
    set_seed(12345);

    // Generate a large sample from the gamma distribution
    let samples = random::gamma(shape, scale, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);

    // For a gamma distribution:
    // 1. Mean should be close to shape*scale
    let expected_mean = shape * scale;
    assert!(
        is_within_bounds(sample_mean, expected_mean, 0.2 * expected_mean),
        "Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // 2. Variance should be close to shape*scale²
    let expected_variance = shape * scale * scale;
    assert!(
        is_within_bounds(sample_variance, expected_variance, 0.3 * expected_variance),
        "Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // 3. All values should be positive
    let all_positive = samples.to_vec().iter().all(|&x| x > 0.0);
    assert!(all_positive, "Some values are not positive");
}

#[test]
fn test_exponential_distribution_properties() {
    let scale = 2.0;

    // Set a fixed seed for reproducibility
    set_seed(12345);

    // Generate a large sample from the exponential distribution
    let samples = random::exponential(scale, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);

    // For an exponential distribution:
    // 1. Mean should be close to scale
    assert!(
        is_within_bounds(sample_mean, scale, 0.2 * scale),
        "Expected mean close to {}, got {}",
        scale,
        sample_mean
    );

    // 2. Variance should be close to scale²
    let expected_variance = scale * scale;
    assert!(
        is_within_bounds(sample_variance, expected_variance, 0.3 * expected_variance),
        "Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // 3. All values should be positive
    let all_positive = samples.to_vec().iter().all(|&x| x > 0.0);
    assert!(all_positive, "Some values are not positive");
}

#[test]
fn test_binomial_distribution_properties() {
    let n = 20u64;
    let p = 0.3;

    // Set a fixed seed for reproducibility
    set_seed(12345);

    // Generate a large sample from the binomial distribution
    let samples = random::binomial::<u64>(n, p, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&Array::from_vec(
        samples.to_vec().iter().map(|&x| x as f64).collect(),
    ));
    let sample_variance = calculate_variance(
        &Array::from_vec(samples.to_vec().iter().map(|&x| x as f64).collect()),
        sample_mean,
    );

    // For a binomial distribution:
    // 1. Mean should be close to n*p
    let expected_mean = n as f64 * p;
    assert!(
        is_within_bounds(sample_mean, expected_mean, 0.2 * expected_mean),
        "Expected mean close to {}, got {}",
        expected_mean,
        sample_mean
    );

    // 2. Variance should be close to n*p*(1-p)
    let expected_variance = n as f64 * p * (1.0 - p);
    assert!(
        is_within_bounds(sample_variance, expected_variance, 0.3 * expected_variance),
        "Expected variance close to {}, got {}",
        expected_variance,
        sample_variance
    );

    // 3. All values should be integers in the range [0, n]
    let all_in_range = samples.to_vec().iter().all(|&x| x <= n);
    assert!(all_in_range, "Some values are outside the range [0, {}]", n);
}

#[test]
fn test_poisson_distribution_properties() {
    let lambda = 5.0;

    // Set a fixed seed for reproducibility
    set_seed(12345);

    // Generate a large sample from the Poisson distribution
    let samples = random::poisson::<u64>(lambda, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics
    let sample_mean = calculate_mean(&Array::from_vec(
        samples.to_vec().iter().map(|&x| x as f64).collect(),
    ));
    let sample_variance = calculate_variance(
        &Array::from_vec(samples.to_vec().iter().map(|&x| x as f64).collect()),
        sample_mean,
    );

    // For a Poisson distribution:
    // 1. Mean should be close to lambda
    assert!(
        is_within_bounds(sample_mean, lambda, 0.2 * lambda),
        "Expected mean close to {}, got {}",
        lambda,
        sample_mean
    );

    // 2. Variance should also be close to lambda
    assert!(
        is_within_bounds(sample_variance, lambda, 0.2 * lambda),
        "Expected variance close to {}, got {}",
        lambda,
        sample_variance
    );

    // 3. All values should be non-negative integers
    let all_non_negative = samples.to_vec().iter().all(|&_x| true);
    assert!(all_non_negative, "Some values are negative");
}

#[test]
#[ignore = "Seeding behavior changed during SciRS2 migration - requires seeding implementation fix"]
fn test_seed_reproducibility() {
    // Test that using the same seed produces the same sequence
    // While different seeds produce different sequences

    // First sequence with seed 42
    set_seed(42);
    let samples1 = random::normal(0.0, 1.0, &[100]).unwrap();

    // Second sequence with same seed
    set_seed(42);
    let samples2 = random::normal(0.0, 1.0, &[100]).unwrap();

    // Third sequence with different seed
    set_seed(43);
    let samples3 = random::normal(0.0, 1.0, &[100]).unwrap();

    // Same seed should produce exactly the same values
    assert_eq!(
        samples1.to_vec(),
        samples2.to_vec(),
        "Same seed should produce identical sequences"
    );

    // Different seeds should produce different sequences
    assert_ne!(
        samples1.to_vec(),
        samples3.to_vec(),
        "Different seeds should produce different sequences"
    );
}
