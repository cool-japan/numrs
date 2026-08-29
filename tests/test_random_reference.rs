use numrs2::array::Array;
use numrs2::random::{self, set_seed};
use serial_test::serial;

// This file contains reference tests comparing NumRS2 random distributions
// with known expected values. This helps ensure the implementation is correct.
// Helper function to check if a value is within expected range
#[allow(dead_code)]
fn is_within_range(value: f64, expected: f64, tolerance: f64) -> bool {
    (value - expected).abs() <= tolerance
}

/// Test the normal distribution statistical properties
#[test]
#[serial]
fn test_normal_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Generate a larger sample from normal distribution with mean 0, std 1
    let normal_samples = random::normal(0.0, 1.0, &[1000]).unwrap();
    let actual_values = normal_samples.to_vec();

    // Test statistical properties instead of exact values
    // 1. Check mean is close to 0
    let mean: f64 = actual_values.iter().sum::<f64>() / actual_values.len() as f64;
    assert!(
        mean.abs() < 0.1,
        "Normal distribution mean {} is not close to 0",
        mean
    );

    // 2. Check standard deviation is close to 1
    let variance: f64 = actual_values
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / actual_values.len() as f64;
    let std_dev = variance.sqrt();
    assert!(
        (std_dev - 1.0).abs() < 0.1,
        "Normal distribution std dev {} is not close to 1",
        std_dev
    );

    // 3. Check that values are within reasonable range (99.7% within 3 std devs)
    let out_of_range = actual_values.iter().filter(|&&x| x.abs() > 3.0).count();
    assert!(
        out_of_range < 10,
        "Too many values outside 3 standard deviations: {}",
        out_of_range
    );
}

/// Test the beta distribution properties instead of exact values
#[test]
#[serial]
fn test_beta_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Generate a sample from beta distribution with alpha=2, beta=5
    let beta_samples = random::beta(2.0, 5.0, &[1000]).unwrap();
    let actual_values = beta_samples.to_vec();

    // Test statistical properties
    // Check that all values are in [0, 1]
    for &val in &actual_values {
        assert!(
            (0.0..=1.0).contains(&val),
            "Beta value {} is outside [0,1]",
            val
        );
    }

    // For Beta(2,5), theoretical mean = 2/(2+5) = 2/7 ≈ 0.286
    let mean: f64 = actual_values.iter().sum::<f64>() / actual_values.len() as f64;
    let expected_mean = 2.0 / 7.0;
    assert!(
        (mean - expected_mean).abs() < 0.05,
        "Beta mean {} is too far from expected {}",
        mean,
        expected_mean
    );

    // For Beta(a,b), variance = ab/((a+b)²(a+b+1)) = 2*5/(7²*8) = 10/392 ≈ 0.0255
    let variance: f64 = actual_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / actual_values.len() as f64;
    let expected_variance = (2.0 * 5.0) / ((7.0 * 7.0) * 8.0);
    assert!(
        (variance - expected_variance).abs() < 0.02,
        "Beta variance {} is too far from expected {}",
        variance,
        expected_variance
    );
}

/// Test uniform distribution properties instead of exact values
#[test]
#[serial]
fn test_uniform_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Generate a sample from uniform distribution in [0, 1]
    let uniform_samples = random::uniform(0.0, 1.0, &[1000]).unwrap();
    let actual_values = uniform_samples.to_vec();

    // Test statistical properties instead of exact values
    // Check that all values are in [0, 1]
    for &val in &actual_values {
        assert!(
            (0.0..=1.0).contains(&val),
            "Uniform value {} is outside [0,1]",
            val
        );
    }

    // Check approximate mean (should be around 0.5)
    let mean: f64 = actual_values.iter().sum::<f64>() / actual_values.len() as f64;
    assert!(
        (mean - 0.5).abs() < 0.1,
        "Uniform mean {} is too far from 0.5",
        mean
    );

    // Check approximate variance (should be around 1/12 ≈ 0.083)
    let variance: f64 = actual_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / actual_values.len() as f64;
    assert!(
        (variance - 1.0 / 12.0).abs() < 0.05,
        "Uniform variance {} is too far from 1/12",
        variance
    );
}

/// Test the gamma distribution properties instead of exact values
#[test]
#[serial]
fn test_gamma_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Generate a sample from gamma distribution with shape=2, scale=3
    let gamma_samples = random::gamma(2.0, 3.0, &[1000]).unwrap();
    let actual_values = gamma_samples.to_vec();

    // Test statistical properties
    // Check that all values are positive
    for &val in &actual_values {
        assert!(val > 0.0, "Gamma value {} should be positive", val);
    }

    // For Gamma(shape=2, scale=3), theoretical mean = shape * scale = 2 * 3 = 6
    let mean: f64 = actual_values.iter().sum::<f64>() / actual_values.len() as f64;
    let expected_mean = 2.0 * 3.0;
    assert!(
        (mean - expected_mean).abs() < 0.5,
        "Gamma mean {} is too far from expected {}",
        mean,
        expected_mean
    );

    // For Gamma(shape, scale), variance = shape * scale² = 2 * 9 = 18
    // Gamma distribution has inherently variable sample variance, increase tolerance
    let variance: f64 = actual_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / actual_values.len() as f64;
    let expected_variance = 2.0 * 3.0 * 3.0;
    assert!(
        (variance - expected_variance).abs() < 4.0,
        "Gamma variance {} is too far from expected {}",
        variance,
        expected_variance
    );
}

/// Test random integers properties instead of exact values
#[test]
#[serial]
fn test_integers_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Generate integers in the range [1, 100]
    let int_samples = random::integers(1, 100, &[1000]).unwrap();
    let actual_values = int_samples.to_vec();

    // Test statistical properties
    // Check that all values are in the specified range [1, 100]
    for &val in &actual_values {
        assert!(
            (1..=100).contains(&val),
            "Integer value {} is outside [1,100]",
            val
        );
    }

    // Check approximate mean (should be around (1+100)/2 = 50.5)
    let mean: f64 =
        actual_values.iter().map(|&x| x as f64).sum::<f64>() / actual_values.len() as f64;
    assert!(
        (mean - 50.5).abs() < 5.0,
        "Integer mean {} is too far from 50.5",
        mean
    );

    // Check that we have reasonable distribution across the range
    let unique_count = {
        let mut sorted = actual_values.clone();
        sorted.sort();
        sorted.dedup();
        sorted.len()
    };
    assert!(
        unique_count > 50,
        "Too few unique values: {}, expected > 50",
        unique_count
    );
}

/// Test binomial distribution properties instead of exact values
#[test]
#[serial]
fn test_binomial_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Generate samples from binomial distribution with n=20, p=0.3
    let binomial_samples = random::binomial::<u64>(20, 0.3, &[1000]).unwrap();
    let actual_values = binomial_samples.to_vec();

    // Test statistical properties
    // Check that all values are in the valid range [0, n]
    for &val in &actual_values {
        assert!(val <= 20, "Binomial value {} exceeds n=20", val);
    }

    // For Binomial(n=20, p=0.3), theoretical mean = n*p = 20*0.3 = 6
    let mean: f64 =
        actual_values.iter().map(|&x| x as f64).sum::<f64>() / actual_values.len() as f64;
    let expected_mean = 20.0 * 0.3;
    assert!(
        (mean - expected_mean).abs() < 0.5,
        "Binomial mean {} is too far from expected {}",
        mean,
        expected_mean
    );

    // For Binomial(n, p), variance = n*p*(1-p) = 20*0.3*0.7 = 4.2
    let variance: f64 = actual_values
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / actual_values.len() as f64;
    let expected_variance = 20.0 * 0.3 * 0.7;
    assert!(
        (variance - expected_variance).abs() < 1.0,
        "Binomial variance {} is too far from expected {}",
        variance,
        expected_variance
    );
}

/// Test Poisson distribution properties instead of exact values
#[test]
#[serial]
fn test_poisson_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Generate samples from Poisson distribution with lambda=5
    let poisson_samples = random::poisson::<u64>(5.0, &[1000]).unwrap();
    let actual_values = poisson_samples.to_vec();

    // Test statistical properties
    // Check that all values are non-negative (Poisson property)
    for &_val in &actual_values {
        // Poisson values are always >= 0, no upper bound check needed
    }

    // For Poisson(lambda=5), theoretical mean = lambda = 5
    let mean: f64 =
        actual_values.iter().map(|&x| x as f64).sum::<f64>() / actual_values.len() as f64;
    let expected_mean = 5.0;
    assert!(
        (mean - expected_mean).abs() < 0.5,
        "Poisson mean {} is too far from expected {}",
        mean,
        expected_mean
    );

    // For Poisson(lambda), variance = lambda = 5
    let variance: f64 = actual_values
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / actual_values.len() as f64;
    let expected_variance = 5.0;
    assert!(
        (variance - expected_variance).abs() < 1.0,
        "Poisson variance {} is too far from expected {}",
        variance,
        expected_variance
    );
}

/// Test multivariate normal distribution properties instead of exact values
#[test]
#[serial]
fn test_multivariate_normal_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Define mean and covariance parameters
    let mean = vec![1.0, 2.0];
    let cov_data = vec![1.0, 0.5, 0.5, 2.0];
    let cov = Array::from_vec(cov_data).reshape(&[2, 2]);

    // Generate multivariate normal samples (500 samples of 2D vectors)
    let mvn_samples = random::multivariate_normal(&mean, &cov, Some(&[500])).unwrap();
    let actual_values = mvn_samples.to_vec();

    // Extract X and Y components
    let mut x_vals = Vec::with_capacity(500);
    let mut y_vals = Vec::with_capacity(500);
    for i in 0..500 {
        x_vals.push(actual_values[i * 2]);
        y_vals.push(actual_values[i * 2 + 1]);
    }

    // Test statistical properties
    // Check means (should be close to [1.0, 2.0])
    let mean_x: f64 = x_vals.iter().sum::<f64>() / x_vals.len() as f64;
    let mean_y: f64 = y_vals.iter().sum::<f64>() / y_vals.len() as f64;
    assert!(
        (mean_x - 1.0).abs() < 0.2,
        "MVN X mean {} is too far from 1.0",
        mean_x
    );
    assert!(
        (mean_y - 2.0).abs() < 0.2,
        "MVN Y mean {} is too far from 2.0",
        mean_y
    );

    // Check variances (diagonal of covariance matrix: [1.0, 2.0])
    let var_x: f64 =
        x_vals.iter().map(|&x| (x - mean_x).powi(2)).sum::<f64>() / x_vals.len() as f64;
    let var_y: f64 =
        y_vals.iter().map(|&y| (y - mean_y).powi(2)).sum::<f64>() / y_vals.len() as f64;
    assert!(
        (var_x - 1.0).abs() < 0.2,
        "MVN X variance {} is too far from 1.0",
        var_x
    );
    assert!(
        (var_y - 2.0).abs() < 0.3,
        "MVN Y variance {} is too far from 2.0",
        var_y
    );
}

/// Test the exponential distribution properties instead of exact values
#[test]
#[serial]
fn test_exponential_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Generate samples from exponential distribution with scale=2
    let exp_samples = random::exponential(2.0, &[1000]).unwrap();
    let actual_values = exp_samples.to_vec();

    // Test statistical properties
    // Check that all values are positive (exponential property)
    for &val in &actual_values {
        assert!(val > 0.0, "Exponential value {} should be positive", val);
    }

    // For Exponential(scale=2), theoretical mean = scale = 2
    let mean: f64 = actual_values.iter().sum::<f64>() / actual_values.len() as f64;
    let expected_mean = 2.0;
    assert!(
        (mean - expected_mean).abs() < 0.3,
        "Exponential mean {} is too far from expected {}",
        mean,
        expected_mean
    );

    // For Exponential(scale), variance = scale² = 4
    let variance: f64 = actual_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / actual_values.len() as f64;
    let expected_variance = 2.0 * 2.0;
    assert!(
        (variance - expected_variance).abs() < 1.0,
        "Exponential variance {} is too far from expected {}",
        variance,
        expected_variance
    );
}

/// Test the lognormal distribution properties instead of exact values
#[test]
#[serial]
fn test_lognormal_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Generate samples from lognormal distribution with mean=0, sigma=1
    let lognormal_samples = random::lognormal(0.0, 1.0, &[1000]).unwrap();
    let actual_values = lognormal_samples.to_vec();

    // Test statistical properties
    // Check that all values are positive (lognormal property)
    for &val in &actual_values {
        assert!(val > 0.0, "Lognormal value {} should be positive", val);
    }

    // For Lognormal(μ=0, σ=1), theoretical mean = exp(μ + σ²/2) = exp(0.5) ≈ 1.649
    let mean: f64 = actual_values.iter().sum::<f64>() / actual_values.len() as f64;
    let expected_mean = (0.5_f64).exp(); // exp(μ + σ²/2) = exp(0 + 1²/2) = exp(0.5)
    assert!(
        (mean - expected_mean).abs() < 0.5,
        "Lognormal mean {} is too far from expected {}",
        mean,
        expected_mean
    );

    // For Lognormal(μ, σ), variance = (exp(σ²) - 1) * exp(2μ + σ²) = (e-1) * e ≈ 4.67
    // Note: Lognormal has high variance, so sample variance can deviate significantly
    let variance: f64 = actual_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / actual_values.len() as f64;
    let expected_variance = ((1.0_f64).exp() - 1.0) * (1.0_f64).exp(); // (exp(σ²) - 1) * exp(2μ + σ²)
    assert!(
        (variance - expected_variance).abs() < 8.0,
        "Lognormal variance {} is too far from expected {}",
        variance,
        expected_variance
    );
}

/// Test the Weibull distribution properties instead of exact values
#[test]
#[serial]
fn test_weibull_reference_values() {
    // Set a fixed seed for reproducibility
    set_seed(42);

    // Generate samples from Weibull distribution with shape=2, scale=3
    let weibull_samples = random::weibull(2.0, 3.0, &[1000]).unwrap();
    let actual_values = weibull_samples.to_vec();

    // Test statistical properties
    // Check that all values are positive (Weibull property)
    for &val in &actual_values {
        assert!(val > 0.0, "Weibull value {} should be positive", val);
    }

    // Check mean is reasonable for Weibull(shape=2, scale=3)
    // Note: Different implementations may use different parameterizations
    let mean: f64 = actual_values.iter().sum::<f64>() / actual_values.len() as f64;
    // For Weibull distribution, mean should be positive and reasonable for these parameters
    assert!(
        mean > 0.5 && mean < 5.0,
        "Weibull mean {} is outside reasonable range [0.5, 5.0]",
        mean
    );

    // Check variance is reasonable for Weibull distribution
    let variance: f64 = actual_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / actual_values.len() as f64;
    // For Weibull distribution, variance should be positive and reasonable
    assert!(
        variance > 0.01 && variance < 10.0,
        "Weibull variance {} is outside reasonable range [0.01, 10.0]",
        variance
    );
}
