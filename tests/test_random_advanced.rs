use numrs2::array::Array;
use numrs2::random::{self, set_seed};

/// This file contains more advanced property-based tests for the random module,
/// focusing on:
/// 1. Testing relationships between distributions
/// 2. Testing distribution-specific properties
/// 3. Testing edge cases and parameter boundaries
const SAMPLE_SIZE: usize = 5000;
const _CONFIDENCE_LEVEL: f64 = 0.05; // 95% confidence

// Helper function to calculate the percentile of a sorted array
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    let idx = (sorted_data.len() as f64 * p).floor() as usize;
    let idx = std::cmp::min(idx, sorted_data.len() - 1);
    sorted_data[idx]
}

// Helper function to calculate the Kolmogorov-Smirnov statistic
// between empirical CDF and a reference CDF function
fn ks_statistic<F>(data: &[f64], cdf: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut data = data.to_vec();
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = data.len() as f64;
    let mut max_diff = 0.0;

    for (i, &x) in data.iter().enumerate() {
        let ecdf = (i + 1) as f64 / n;
        let theo_cdf = cdf(x);
        let diff = (ecdf - theo_cdf).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    max_diff
}

// Function to test if a sample follows the standard normal distribution
fn test_standard_normal_conformance(samples: &Array<f64>) -> bool {
    let data = samples.to_vec();

    // Standard normal CDF (approximation using the error function approximation)
    // This avoids using the unstable erf() function
    let normal_cdf = |x: f64| -> f64 {
        if x < -10.0 {
            return 0.0;
        }
        if x > 10.0 {
            return 1.0;
        }

        // Approximation of error function
        let erf_approx = |z: f64| -> f64 {
            // Constants for Abramowitz and Stegun approximation 7.1.26
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let p = 0.3275911;

            let sign = if z < 0.0 { -1.0 } else { 1.0 };
            let z = z.abs();

            let t = 1.0 / (1.0 + p * z);
            let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-z * z).exp();

            sign * y
        };

        0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
    };

    let ks = ks_statistic(&data, normal_cdf);

    // Critical value for KS test (95% confidence)
    let critical_value = 1.36 / (SAMPLE_SIZE as f64).sqrt();

    ks <= critical_value
}

#[test]
fn test_normal_to_chi_square_relationship() {
    // A sum of k independent squared standard normal variables
    // follows a chi-square distribution with k degrees of freedom

    let k = 4;
    set_seed(12345);

    // Generate k sets of standard normal variables
    let mut chi_square_samples = Vec::with_capacity(SAMPLE_SIZE);

    for _ in 0..SAMPLE_SIZE {
        let normal_samples = random::standard_normal::<f64>(&[k]).unwrap();
        let sum_of_squares: f64 = normal_samples.to_vec().iter().map(|x| x * x).sum();

        chi_square_samples.push(sum_of_squares);
    }

    // Convert to array for analysis
    let samples = Array::from_vec(chi_square_samples);

    // Calculate mean and variance
    let data = samples.to_vec();
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

    // For a chi-square distribution with k degrees of freedom:
    // Mean = k, Variance = 2k
    let k_f64 = k as f64;
    assert!(
        (mean - k_f64).abs() <= 0.3 * k_f64,
        "Expected mean close to {}, got {}",
        k_f64,
        mean
    );
    assert!(
        (variance - 2.0 * k_f64).abs() <= 0.5 * k_f64,
        "Expected variance close to {}, got {}",
        2.0 * k_f64,
        variance
    );
}

#[test]
fn test_gamma_to_exponential_relationship() {
    // An exponential distribution with rate parameter λ
    // is a special case of gamma distribution with shape=1 and scale=1/λ

    let rate = 0.5; // λ
    let scale = 1.0 / rate; // exponential scale = 1/λ

    set_seed(12345);

    // Generate samples from exponential distribution
    let exp_samples = random::exponential(scale, &[SAMPLE_SIZE]).unwrap();

    // Generate samples from gamma distribution with shape=1 and same scale
    let gamma_samples = random::gamma(1.0, scale, &[SAMPLE_SIZE]).unwrap();

    // Calculate statistics for comparison
    let exp_data = exp_samples.to_vec();
    let gamma_data = gamma_samples.to_vec();

    // Sort data for percentile calculation
    let mut sorted_exp = exp_data.clone();
    sorted_exp.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sorted_gamma = gamma_data.clone();
    sorted_gamma.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compare percentiles
    for p in [0.1, 0.25, 0.5, 0.75, 0.9].iter() {
        let exp_percentile = percentile(&sorted_exp, *p);
        let gamma_percentile = percentile(&sorted_gamma, *p);

        // The percentiles should be similar
        assert!(
            (exp_percentile - gamma_percentile).abs() <= 0.2 * exp_percentile,
            "{}th percentile differs too much: exp={}, gamma={}",
            p * 100.0,
            exp_percentile,
            gamma_percentile
        );
    }
}

#[test]
fn test_binomial_to_normal_approximation() {
    // For large n and p not too close to 0 or 1,
    // binomial(n,p) ≈ normal(np, np(1-p))

    let n = 100u64;
    let p = 0.4;

    set_seed(12345);

    // Generate binomial samples
    let binomial_samples = random::binomial::<u64>(n, p, &[SAMPLE_SIZE]).unwrap();

    // Convert to f64 for comparison
    let binomial_f64 = Array::from_vec(
        binomial_samples
            .to_vec()
            .iter()
            .map(|&x| x as f64)
            .collect(),
    );

    // Expected normal parameters
    let mean = n as f64 * p;
    let std_dev = (n as f64 * p * (1.0 - p)).sqrt();

    // Generate comparable normal samples
    set_seed(54321); // Different seed to ensure independence
    let normal_samples = random::normal(mean, std_dev, &[SAMPLE_SIZE]).unwrap();

    // Since the binomial is discrete and normal is continuous,
    // we'll compare the empirical CDFs
    let binomial_data = binomial_f64.to_vec();
    let normal_data = normal_samples.to_vec();

    let mut sorted_binomial = binomial_data.clone();
    sorted_binomial.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sorted_normal = normal_data.clone();
    sorted_normal.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compare several percentiles
    // With large n, the distributions should be similar at these percentiles
    for p in [0.1, 0.25, 0.5, 0.75, 0.9].iter() {
        let binomial_percentile = percentile(&sorted_binomial, *p);
        let normal_percentile = percentile(&sorted_normal, *p);

        // The percentiles should be relatively close, considering the approximation
        assert!(
            (binomial_percentile - normal_percentile).abs() <= 1.5,
            "{}th percentile differs too much: binomial={}, normal={}",
            p * 100.0,
            binomial_percentile,
            normal_percentile
        );
    }
}

#[test]
fn test_studentt_to_normal_relation() {
    // As degrees of freedom increase, t-distribution approaches normal distribution

    let high_df = 100.0; // High degrees of freedom

    set_seed(12345);

    // Generate t-distribution with high df
    let t_samples = random::student_t(high_df, &[SAMPLE_SIZE]).unwrap();

    // Generate standard normal samples
    set_seed(54321); // Different seed
    let normal_samples = random::standard_normal::<f64>(&[SAMPLE_SIZE]).unwrap();

    // Test if the t-distribution samples approximately follow a standard normal
    let t_data = t_samples.to_vec();
    let mut sorted_t = t_data.clone();
    sorted_t.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let normal_data = normal_samples.to_vec();
    let mut sorted_normal = normal_data.clone();
    sorted_normal.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compare several percentiles
    for p in [0.1, 0.25, 0.5, 0.75, 0.9].iter() {
        let t_percentile = percentile(&sorted_t, *p);
        let normal_percentile = percentile(&sorted_normal, *p);

        // With high df, percentiles should be close
        assert!(
            (t_percentile - normal_percentile).abs() <= 0.2,
            "{}th percentile differs too much: t={}, normal={}",
            p * 100.0,
            t_percentile,
            normal_percentile
        );
    }
}

#[test]
fn test_uniform_sum_to_approximate_normal() {
    // By the Central Limit Theorem, sums of uniform random variables
    // approximate a normal distribution

    let num_uniforms = 12; // Sum of 12 uniform(0,1) is close to normal(6,1)

    set_seed(12345);

    // Generate samples as sums of uniform(0,1)
    let mut sum_samples = Vec::with_capacity(SAMPLE_SIZE);

    for _ in 0..SAMPLE_SIZE {
        let uniform_samples = random::uniform(0.0, 1.0, &[num_uniforms]).unwrap();
        let sum: f64 = uniform_samples.to_vec().iter().sum();

        sum_samples.push(sum);
    }

    let samples = Array::from_vec(sum_samples);

    // For sum of n uniforms (0,1): mean=n/2, variance=n/12
    let expected_mean = num_uniforms as f64 / 2.0;
    let expected_variance = num_uniforms as f64 / 12.0;

    // Calculate actual mean and variance
    let data = samples.to_vec();
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

    // Verify mean and variance
    assert!(
        (mean - expected_mean).abs() <= 0.1,
        "Expected mean close to {}, got {}",
        expected_mean,
        mean
    );
    assert!(
        (variance - expected_variance).abs() <= 0.1,
        "Expected variance close to {}, got {}",
        expected_variance,
        variance
    );

    // Standardize the data
    let standardized = samples
        .to_vec()
        .iter()
        .map(|x| (x - expected_mean) / expected_variance.sqrt())
        .collect::<Vec<f64>>();

    // Test if the standardized sum follows standard normal
    let standardized_array = Array::from_vec(standardized);
    assert!(
        test_standard_normal_conformance(&standardized_array),
        "Sum of uniform variables should approximate normal distribution"
    );
}

#[test]
fn test_random_distribution_edge_cases() {
    // Test edge cases for various distributions

    // 1. Beta distribution with alpha=beta=1 should be uniform(0,1)
    set_seed(12345);
    let beta_samples = random::beta(1.0, 1.0, &[SAMPLE_SIZE]).unwrap();
    let uniform_samples = random::uniform(0.0, 1.0, &[SAMPLE_SIZE]).unwrap();

    // Compare the percentiles
    let beta_data = beta_samples.to_vec();
    let uniform_data = uniform_samples.to_vec();

    let mut sorted_beta = beta_data.clone();
    sorted_beta.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sorted_uniform = uniform_data.clone();
    sorted_uniform.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for p in [0.1, 0.25, 0.5, 0.75, 0.9].iter() {
        let beta_percentile = percentile(&sorted_beta, *p);
        let uniform_percentile = percentile(&sorted_uniform, *p);

        // Beta(1,1) should be similar to Uniform(0,1)
        assert!(
            (beta_percentile - uniform_percentile).abs() <= 0.1,
            "{}th percentile differs too much: beta(1,1)={}, uniform(0,1)={}",
            p * 100.0,
            beta_percentile,
            uniform_percentile
        );
    }

    // 2. Gamma shape=1 should be exponential
    set_seed(12345);
    let scale = 2.0;
    let gamma_samples = random::gamma(1.0, scale, &[SAMPLE_SIZE]).unwrap();
    let exp_samples = random::exponential(scale, &[SAMPLE_SIZE]).unwrap();

    let gamma_mean = gamma_samples.to_vec().iter().sum::<f64>() / SAMPLE_SIZE as f64;
    let exp_mean = exp_samples.to_vec().iter().sum::<f64>() / SAMPLE_SIZE as f64;

    assert!(
        (gamma_mean - exp_mean).abs() <= 0.3 * scale,
        "Gamma(1,s) and Exponential(s) should have similar means"
    );
}

#[test]
fn test_distribution_independence() {
    // Test that samples from different distributions are independent
    // by checking for correlation between pairs

    let n = 1000;
    set_seed(12345);

    // Generate samples from different distributions
    let normal_samples = random::normal(0.0, 1.0, &[n]).unwrap();
    let uniform_samples = random::uniform(0.0, 1.0, &[n]).unwrap();
    let gamma_samples = random::gamma(2.0, 1.0, &[n]).unwrap();
    let exp_samples = random::exponential(1.0, &[n]).unwrap();

    // Calculate correlation
    fn correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let x_mean = x.iter().sum::<f64>() / n;
        let y_mean = y.iter().sum::<f64>() / n;

        let numerator = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum::<f64>();

        let x_variance = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum::<f64>();

        let y_variance = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>();

        numerator / (x_variance.sqrt() * y_variance.sqrt())
    }

    // Check correlations between pairs
    let corr_normal_uniform = correlation(&normal_samples.to_vec(), &uniform_samples.to_vec());

    let corr_normal_gamma = correlation(&normal_samples.to_vec(), &gamma_samples.to_vec());

    let corr_uniform_exp = correlation(&uniform_samples.to_vec(), &exp_samples.to_vec());

    // All correlations should be close to zero
    // Using a relatively high threshold due to finite sample size
    assert!(
        corr_normal_uniform.abs() <= 0.1,
        "Normal and uniform distributions should be uncorrelated"
    );
    assert!(
        corr_normal_gamma.abs() <= 0.1,
        "Normal and gamma distributions should be uncorrelated"
    );
    assert!(
        corr_uniform_exp.abs() <= 0.1,
        "Uniform and exponential distributions should be uncorrelated"
    );
}
