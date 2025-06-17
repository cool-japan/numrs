//! Tests for the SciRS2 integration functionality
//!
//! This file contains tests that verify the correct integration between
//! NumRS2 and SciRS2, particularly for advanced statistical distributions.
//! These tests only run when the "scirs" feature is enabled.
//!
//! Note: This file has been updated to match the SciRS2 v0.1.0-alpha.4 module structure
//! and class names (NoncentralChiSquared, FNoncentral, MaxwellBoltzmann, etc.).

#![cfg(feature = "scirs")]

use approx::assert_relative_eq;
use numrs2::array::Array;
use numrs2::interop::scirs_compat::*;
use numrs2::random::distributions::{set_seed, multivariate_normal_with_rotation};
use numrs2::random::advanced_distributions::{vonmises, maxwell};
use numrs2::random::distributions_enhanced::truncated_normal;
use std::f64::consts::PI;

/// Test that the noncentral chi-square distribution generates valid values and
/// respects the expected statistical properties
#[test]
fn test_noncentral_chisquare() {
    // Set a seed for reproducibility
    set_seed(12345);

    // Test with valid parameters
    let df = 5.0f64;
    let nonc = 2.0f64;
    let samples = noncentral_chisquare(df, nonc, &[1000]).unwrap();

    // Check shape
    assert_eq!(samples.shape(), vec![1000]);

    // All samples should be positive
    for val in samples.to_vec() {
        assert!(val > 0.0);
    }

    // Check basic statistical properties
    let data = samples.to_vec();
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let expected_mean = df + nonc;

    // The mean should be approximately df + nonc
    // Using a loose tolerance due to sampling variance
    assert_relative_eq!(mean, expected_mean, epsilon = 0.5);

    // Test with invalid parameters
    assert!(noncentral_chisquare(-1.0, 2.0, &[10]).is_err());
    assert!(noncentral_chisquare(2.0, -1.0, &[10]).is_err());
}

/// Test that the noncentral F distribution generates valid values
#[test]
fn test_noncentral_f() {
    // Set a seed for reproducibility
    set_seed(12345);

    // Test with valid parameters
    let dfnum = 5.0f64;
    let dfden = 10.0f64;
    let nonc = 2.0f64;
    let samples = noncentral_f(dfnum, dfden, nonc, &[100]).unwrap();

    // Check shape
    assert_eq!(samples.shape(), vec![100]);

    // All samples should be positive
    for val in samples.to_vec() {
        assert!(val > 0.0);
    }

    // Test with invalid parameters
    assert!(noncentral_f(-1.0, 10.0, 2.0, &[10]).is_err());
    assert!(noncentral_f(5.0, -1.0, 2.0, &[10]).is_err());
    assert!(noncentral_f(5.0, 10.0, -1.0, &[10]).is_err());
}

/// Test that the von Mises distribution generates valid values
#[test]
fn test_vonmises() {
    // Set a seed for reproducibility
    set_seed(12345);

    // Test with valid parameters
    let mu = 0.0f64;
    let kappa = 2.0f64;
    let samples = vonmises(mu, kappa, &[500]).unwrap();

    // Check shape
    assert_eq!(samples.shape(), vec![500]);

    // All samples should be in range [-PI, PI]
    for val in samples.to_vec() {
        assert!(val >= -PI && val <= PI);
    }

    // Check that the samples are concentrated around mu
    let data = samples.to_vec();
    let mean_direction = data.iter().map(|&x| x).collect::<Vec<f64>>();
    let mean_sin = mean_direction.iter().map(|&x| x.sin()).sum::<f64>() / data.len() as f64;
    let mean_cos = mean_direction.iter().map(|&x| x.cos()).sum::<f64>() / data.len() as f64;
    let circular_mean = mean_sin.atan2(mean_cos);

    // For mu = 0, the circular mean should be close to 0
    assert_relative_eq!(circular_mean, mu, epsilon = 0.2);

    // Test with invalid parameters
    assert!(vonmises(0.0, -1.0, &[10]).is_err());
}

/// Test that the Maxwell distribution generates valid values
#[test]
fn test_maxwell() {
    // Set a seed for reproducibility
    set_seed(12345);

    // Test with valid parameters
    let scale = 1.0f64;
    let samples = maxwell(scale, &[500]).unwrap();

    // Check shape
    assert_eq!(samples.shape(), vec![500]);

    // All samples should be positive
    for val in samples.to_vec() {
        assert!(val > 0.0);
    }

    // Check statistical properties
    let data = samples.to_vec();
    let mean = data.iter().sum::<f64>() / data.len() as f64;

    // The mean of a Maxwell distribution with scale=1 is approximately 1.5957...
    let expected_mean = 2.0 * scale * (2.0 / PI).sqrt();
    assert_relative_eq!(mean, expected_mean, epsilon = 0.1);

    // Test with invalid parameters
    assert!(maxwell(-1.0, &[10]).is_err());
    assert!(maxwell(0.0, &[10]).is_err());
}

/// Test that the truncated normal distribution generates valid values
#[test]
fn test_truncated_normal() {
    // Set a seed for reproducibility
    set_seed(12345);

    // Test with valid parameters
    let mean = 0.0f64;
    let std = 1.0f64;
    let low = -2.0f64;
    let high = 2.0f64;
    let samples = truncated_normal(mean, std, low, high, &[500]).unwrap();

    // Check shape
    assert_eq!(samples.shape(), vec![500]);

    // All samples should be within bounds
    for val in samples.to_vec() {
        assert!(val >= low && val <= high);
    }

    // Test with invalid parameters
    assert!(truncated_normal(0.0, -1.0, -2.0, 2.0, &[10]).is_err());
    assert!(truncated_normal(0.0, 1.0, 2.0, -2.0, &[10]).is_err());
    assert!(truncated_normal(0.0, 1.0, 2.0, 2.0, &[10]).is_err());
}

/// Test that the multivariate normal with rotation distribution generates valid values
#[test]
fn test_multivariate_normal_with_rotation() {
    // Set a seed for reproducibility
    set_seed(12345);

    // Test with valid parameters
    let mean = vec![0.0f64, 0.0f64];
    let cov_data = vec![1.0, 0.5, 0.5, 1.0];
    let cov = Array::from_vec(cov_data).reshape(&[2, 2]);

    // Without rotation
    let samples = multivariate_normal_with_rotation(&mean, &cov, Some(&[100]), None).unwrap();

    // Check shape
    assert_eq!(samples.shape(), vec![100, 2]);

    // Test with rotation matrix
    let rot_data = vec![0.7071, 0.7071, -0.7071, 0.7071]; // 45-degree rotation matrix
    let rotation = Array::from_vec(rot_data).reshape(&[2, 2]);

    let samples_rotated =
        multivariate_normal_with_rotation(&mean, &cov, Some(&[100]), Some(&rotation)).unwrap();

    // Check shape
    assert_eq!(samples_rotated.shape(), vec![100, 2]);

    // Test with invalid parameters
    // Empty mean vector
    assert!(multivariate_normal_with_rotation(&[], &cov, Some(&[10]), None).is_err());

    // Mismatched covariance dimension
    let bad_cov = Array::from_vec(vec![1.0, 0.5, 0.5]).reshape(&[3, 1]);
    assert!(multivariate_normal_with_rotation(&mean, &bad_cov, Some(&[10]), None).is_err());
}

/// Test generating arrays of different shapes with SciRS2 distributions
#[test]
fn test_distribution_shapes() {
    // Set a seed for reproducibility
    set_seed(12345);

    // Test 1D array
    let samples_1d = noncentral_chisquare(5.0f64, 2.0f64, &[10]).unwrap();
    assert_eq!(samples_1d.shape(), vec![10]);

    // Test 2D array
    let samples_2d = noncentral_chisquare(5.0f64, 2.0f64, &[5, 4]).unwrap();
    assert_eq!(samples_2d.shape(), vec![5, 4]);

    // Test 3D array
    let samples_3d = noncentral_chisquare(5.0f64, 2.0f64, &[2, 3, 4]).unwrap();
    assert_eq!(samples_3d.shape(), vec![2, 3, 4]);
}

/// Test the repeatability of random generation when using the same seed
#[test]
fn test_seed_repeatability() {
    // First run
    set_seed(42);
    let samples1 = vonmises(0.0f64, 1.0f64, &[10]).unwrap();

    // Second run with same seed
    set_seed(42);
    let samples2 = vonmises(0.0f64, 1.0f64, &[10]).unwrap();

    // Third run with different seed
    set_seed(24);
    let samples3 = vonmises(0.0f64, 1.0f64, &[10]).unwrap();

    // The first two should be identical
    assert_eq!(samples1.to_vec(), samples2.to_vec());

    // The third should be different
    assert_ne!(samples1.to_vec(), samples3.to_vec());
}

/// Test the conversion between NumRS2 and SciRS2 types
#[test]
fn test_type_conversions() {
    // Test with f64
    let samples_f64 = maxwell(1.0f64, &[10]).unwrap();
    assert_eq!(samples_f64.shape(), vec![10]);

    // Test with f32
    let samples_f32 = maxwell(1.0f32, &[10]).unwrap();
    assert_eq!(samples_f32.shape(), vec![10]);

    // Both should contain valid values
    for val in samples_f64.to_vec() {
        assert!(val > 0.0);
    }

    for val in samples_f32.to_vec() {
        assert!(val > 0.0);
    }
}
