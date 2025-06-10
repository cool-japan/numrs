//! Reference tests for SciRS2 integration
//!
//! This file contains reference tests that compare the output of NumRS2's SciRS2
//! integration with reference values. These tests verify that our implementation
//! produces results that match expected statistical properties.
//!
//! Note: These tests only run when the "scirs" feature is enabled.

#![cfg(feature = "scirs")]

use approx::assert_relative_eq;
use numrs2::array::Array;
use numrs2::interop::scirs_compat::*;
use numrs2::random::distributions::set_seed;
use std::f64::consts::PI;

/// Utility function to calculate the mean of a sample
fn calculate_mean(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

/// Utility function to calculate the variance of a sample
fn calculate_variance(data: &[f64]) -> f64 {
    let mean = calculate_mean(data);
    let sum_squared_diff = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
    sum_squared_diff / (data.len() - 1) as f64
}

/// Test that noncentral chi-square distribution produces correct statistical properties
#[test]
fn test_noncentral_chisquare_statistics() {
    // Fix seed for reproducibility
    set_seed(12345);

    // Generate a large sample to get stable statistics
    let df = 5.0;
    let nonc = 2.0;
    let samples = noncentral_chisquare(df, nonc, &[10000]).unwrap();
    let data = samples.to_vec();

    // Calculate statistics
    let mean = calculate_mean(&data);
    let variance = calculate_variance(&data);

    // Expected values (based on theoretical properties)
    // Mean = df + nonc
    let expected_mean = df + nonc;
    // Variance = 2*df + 4*nonc
    let expected_variance = 2.0 * df + 4.0 * nonc;

    // Check that our statistics match the expected values (with reasonable tolerance)
    assert_relative_eq!(mean, expected_mean, epsilon = 0.1, max_relative = 0.05);
    assert_relative_eq!(
        variance,
        expected_variance,
        epsilon = 0.5,
        max_relative = 0.1
    );
}

/// Test that noncentral F distribution produces correct statistical properties
#[test]
fn test_noncentral_f_statistics() {
    // Fix seed for reproducibility
    set_seed(12345);

    // Generate a large sample to get stable statistics
    let dfnum = 10.0;
    let dfden = 20.0;
    let nonc = 2.0;
    let samples = noncentral_f(dfnum, dfden, nonc, &[5000]).unwrap();
    let data = samples.to_vec();

    // Calculate statistics
    let mean = calculate_mean(&data);

    // Expected mean for large dfden (approximation)
    // Mean ≈ (dfden / (dfden - 2)) * (dfnum + nonc) / dfnum
    let expected_mean = (dfden / (dfden - 2.0)) * (dfnum + nonc) / dfnum;

    // Check that our mean matches the expected value (with reasonable tolerance)
    // Noncentral F can have high variance, so we use a more permissive epsilon
    assert_relative_eq!(mean, expected_mean, epsilon = 0.15, max_relative = 0.1);
}

/// Test that von Mises distribution produces correct concentration around the mean
#[test]
fn test_vonmises_concentration() {
    // Fix seed for reproducibility
    set_seed(12345);

    // Generate samples with different concentration parameters
    let mu: f64 = 0.0;
    let kappas: [f64; 4] = [0.5, 2.0, 5.0, 10.0];
    let sample_count = 2000;

    for &kappa in &kappas {
        let samples = vonmises(mu, kappa, &[sample_count]).unwrap();
        let data = samples.to_vec();

        // Count samples within PI/6 of the mean (30 degrees)
        let within_range = data.iter().filter(|&&x| (x - mu).abs() <= (PI / 6.0)).count();
        let proportion = within_range as f64 / sample_count as f64;

        // Calculate theoretical proportion using the cumulative distribution function
        // (approximation based on the CDF of von Mises)
        let range = PI / 6.0;
        let theoretical_prop = if kappa < 0.1 {
            // For very small kappa, approximate as uniform distribution
            range / PI
        } else {
            // For larger kappa, use a normal approximation
            // The von Mises distribution approaches a normal distribution with
            // variance 1/kappa for large kappa
            let std_dev = 1.0 / kappa.sqrt();
            0.5 * (erf(range / (std_dev * 1.414213562))) // erf(x/√2) gives proportion within x std deviations
        };

        // Check that our concentration matches the theoretical prediction
        // Allow for sampling variation with a reasonable epsilon
        println!(
            "kappa = {}: observed = {:.4}, theoretical = {:.4}",
            kappa, proportion, theoretical_prop
        );

        // The error should be smaller for larger kappa
        let epsilon = if kappa < 1.0 {
            0.1
        } else if kappa < 5.0 {
            0.08
        } else {
            0.05
        };
        assert!(
            (proportion - theoretical_prop).abs() < epsilon,
            "Proportion difference too large: observed = {:.4}, theoretical = {:.4}, kappa = {}",
            proportion,
            theoretical_prop,
            kappa
        );
    }
}

/// Test that Maxwell-Boltzmann distribution produces correct statistical properties
#[test]
fn test_maxwell_statistics() {
    // Fix seed for reproducibility
    set_seed(12345);

    // Generate samples with different scale parameters
    let scales = [0.5, 1.0, 2.0];
    let sample_count = 5000;

    for &scale in &scales {
        let samples = maxwell(scale, &[sample_count]).unwrap();
        let data = samples.to_vec();

        // Calculate statistics
        let mean = calculate_mean(&data);
        let variance = calculate_variance(&data);

        // Expected values (based on theoretical properties)
        // Mean = 2*scale*sqrt(2/π)
        let expected_mean = 2.0 * scale * (2.0 / PI).sqrt();
        // Variance = scale²*(3*π - 8)/π
        let expected_variance = scale.powi(2) * (3.0 * PI - 8.0) / PI;

        // Check that our statistics match the expected values
        assert_relative_eq!(mean, expected_mean, epsilon = 0.05, max_relative = 0.05);
        assert_relative_eq!(
            variance,
            expected_variance,
            epsilon = 0.1,
            max_relative = 0.1
        );
    }
}

/// Test that truncated normal distribution produces correct results
#[test]
fn test_truncated_normal_statistics() {
    // Fix seed for reproducibility
    set_seed(12345);

    // Generate samples with different truncation ranges
    let test_cases = [
        // (mean, std, low, high, expected_mean)
        (0.0, 1.0, -1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 2.0, 0.8),
        (0.0, 1.0, -2.0, 0.0, -0.8),
        (1.0, 2.0, -1.0, 3.0, 1.0),
    ];

    for &(mean, std, low, high, expected_mean) in &test_cases {
        let samples = truncated_normal(mean, std, low, high, &[5000]).unwrap();
        let data = samples.to_vec();

        // Check that all samples are within bounds
        for &val in &data {
            assert!(val >= low && val <= high);
        }

        let actual_mean = calculate_mean(&data);

        // Check that our mean is close to the expected value
        // The expected values are approximate, so we use a generous epsilon
        assert_relative_eq!(
            actual_mean,
            expected_mean,
            epsilon = 0.1,
            max_relative = 0.1
        );
    }
}

/// Test that multivariate normal with rotation produces correct correlations
#[test]
fn test_multivariate_normal_rotation() {
    // Fix seed for reproducibility
    set_seed(12345);

    // Set up mean and covariance matrix
    let mean = vec![0.0, 0.0];
    let corr = 0.7; // Correlation between variables
    let cov_data = vec![1.0, corr, corr, 1.0];
    let cov = Array::from_vec(cov_data).reshape(&[2, 2]);

    // Generate samples without rotation
    let samples1 = multivariate_normal_with_rotation(&mean, &cov, Some(&[5000]), None).unwrap();
    let data1 = samples1.to_vec();

    // Calculate correlation
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..5000 {
        let x = data1[i * 2];
        let y = data1[i * 2 + 1];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }

    let n = 5000.0;
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let cov_xy = (sum_xy - n * mean_x * mean_y) / (n - 1.0);
    let var_x: f64 = (sum_x2 - n * mean_x * mean_x) / (n - 1.0);
    let var_y: f64 = (sum_y2 - n * mean_y * mean_y) / (n - 1.0);

    let corr_xy = cov_xy / (var_x.sqrt() * var_y.sqrt());

    // Check that the observed correlation is close to the specified correlation
    assert_relative_eq!(corr_xy, corr, epsilon = 0.05);

    // Now test with a rotation matrix (90 degrees)
    let rot_data = vec![0.0, 1.0, -1.0, 0.0];
    let rotation = Array::from_vec(rot_data).reshape(&[2, 2]);

    let samples2 =
        multivariate_normal_with_rotation(&mean, &cov, Some(&[5000]), Some(&rotation)).unwrap();
    let data2 = samples2.to_vec();

    // Calculate correlation for rotated data
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..5000 {
        let x = data2[i * 2];
        let y = data2[i * 2 + 1];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;
    let cov_xy = (sum_xy - n * mean_x * mean_y) / (n - 1.0);
    let var_x: f64 = (sum_x2 - n * mean_x * mean_x) / (n - 1.0);
    let var_y: f64 = (sum_y2 - n * mean_y * mean_y) / (n - 1.0);

    let corr_xy_rotated = cov_xy / (var_x.sqrt() * var_y.sqrt());

    // For a 90-degree rotation, the correlation should be negated
    assert_relative_eq!(corr_xy_rotated, -corr, epsilon = 0.05);
}

/// Error function approximation (used in test_vonmises_concentration)
fn erf(x: f64) -> f64 {
    // Constants for Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}
