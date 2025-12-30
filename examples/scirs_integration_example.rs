//! Example demonstrating the SciRS2 integration in NumRS2
//!
//! This example shows how to use the advanced statistical distributions
//! provided by SciRS2 integration in NumRS2.
//!
//! When available (with the "scirs" feature enabled), these distributions
//! provide additional functionality beyond what's available in the standard
//! NumRS2 distributions.
//!
//! Run this example with:
//! `cargo run --example scirs_integration_example --features scirs`

#![allow(clippy::result_large_err)]

#[cfg(feature = "scirs")]
use numrs2::array::Array;
use numrs2::error::Result;
#[cfg(feature = "scirs")]
use numrs2::random::distributions::multivariate_normal_with_rotation;
use numrs2::random::distributions::set_seed;

// Import the regular distributions for comparison
#[cfg(feature = "scirs")]
use numrs2::random::advanced_distributions::{maxwell, vonmises};
#[cfg(feature = "scirs")]
use numrs2::random::distributions::student_t;
use numrs2::random::distributions::{chisquare, normal};
#[cfg(feature = "scirs")]
use numrs2::random::distributions_enhanced::truncated_normal;

// Import the SciRS2 integration when available
#[cfg(feature = "scirs")]
use numrs2::interop::scirs_compat::*;

fn main() -> Result<()> {
    println!("NumRS2 SciRS2 Integration Example");
    println!("=================================");

    // Set a seed for reproducibility
    println!("\nSetting a seed for reproducibility...");
    set_seed(42);

    // Check if SciRS2 integration is enabled
    println!("\nSciRS2 Feature Status");
    println!("-------------------");

    #[cfg(feature = "scirs")]
    {
        println!("✓ SciRS2 integration is ENABLED");
        println!("  Advanced statistical distributions are available");
    }

    #[cfg(not(feature = "scirs"))]
    {
        println!("✗ SciRS2 integration is DISABLED");
        println!("  Advanced statistical distributions are NOT available");
        println!("  To enable, rebuild with: cargo run --example scirs_integration_example --features scirs");
    }

    println!("\nPart 1: Comparing Native and Advanced Distributions");
    println!("------------------------------------------------");

    // 1. Standard Chi-square vs. Noncentral Chi-square
    println!("\nChi-square Distributions:");

    // Standard chi-square (native NumRS2)
    let chi2_samples = chisquare(3.0, &[5])?;
    println!("- Standard Chi-square (df=3) samples: {:?}", chi2_samples);

    // Noncentral chi-square (via SciRS2 when available)
    #[cfg(feature = "scirs")]
    {
        let nonc_chi2_samples = noncentral_chisquare(3.0, 2.0, &[5])?;
        println!(
            "- Noncentral Chi-square (df=3, nonc=2) samples: {:?}",
            nonc_chi2_samples
        );
        println!(
            "  * Note: The noncentral parameter introduces a rightward shift in the distribution"
        );
    }

    #[cfg(not(feature = "scirs"))]
    {
        println!("- Noncentral Chi-square: Not available (requires SciRS2 integration)");
    }

    // 2. Standard F vs. Noncentral F
    println!("\nF Distributions:");

    // Standard Student t distribution (native NumRS2, when scirs feature is enabled)
    #[cfg(feature = "scirs")]
    {
        let t_samples = student_t(3.0, &[5])?;
        println!("- Student t (df=3) samples: {:?}", t_samples);
    }

    #[cfg(not(feature = "scirs"))]
    {
        println!("- Standard F distribution: Not available (requires SciRS2 integration)");
    }

    // Noncentral F distribution (via SciRS2 when available)
    #[cfg(feature = "scirs")]
    {
        let nonc_f_samples = noncentral_f(3.0, 10.0, 2.0, &[5])?;
        println!(
            "- Noncentral F (dfnum=3, dfden=10, nonc=2) samples: {:?}",
            nonc_f_samples
        );
        println!("  * Note: The noncentral parameter shifts the distribution to the right");
    }

    #[cfg(not(feature = "scirs"))]
    {
        println!("- Noncentral F: Not available (requires SciRS2 integration)");
    }

    println!("\nPart 2: Advanced Distributions from SciRS2");
    println!("----------------------------------------");

    // Von Mises distribution (circular normal)
    println!("\nVon Mises Distribution (Circular Normal):");

    #[cfg(feature = "scirs")]
    {
        // Generate samples with different concentration parameters
        let mu = 0.0; // Mean direction

        let low_kappa_samples = vonmises(mu, 0.5, &[5])?;
        println!(
            "- Von Mises (mu=0, kappa=0.5, low concentration) samples: {:?}",
            low_kappa_samples
        );

        let high_kappa_samples = vonmises(mu, 5.0, &[5])?;
        println!(
            "- Von Mises (mu=0, kappa=5.0, high concentration) samples: {:?}",
            high_kappa_samples
        );
        println!("  * Note: Higher kappa means tighter concentration around the mean direction");
    }

    #[cfg(not(feature = "scirs"))]
    {
        println!("- Von Mises distribution: Not available (requires SciRS2 integration)");
    }

    // Maxwell-Boltzmann distribution
    println!("\nMaxwell-Boltzmann Distribution:");

    #[cfg(feature = "scirs")]
    {
        let maxwell_samples = maxwell(1.0, &[5])?;
        println!("- Maxwell (scale=1.0) samples: {:?}", maxwell_samples);
        println!("  * Note: Used for modeling particle velocities in gases");
    }

    #[cfg(not(feature = "scirs"))]
    {
        println!("- Maxwell-Boltzmann distribution: Not available (requires SciRS2 integration)");
    }

    // Truncated Normal distribution
    println!("\nTruncated Normal Distribution:");

    // Standard normal (native NumRS2)
    let normal_samples = normal(0.0, 1.0, &[5])?;
    println!(
        "- Standard Normal (mean=0, std=1) samples: {:?}",
        normal_samples
    );

    #[cfg(feature = "scirs")]
    {
        // Truncated normal (via SciRS2)
        let trunc_normal_samples = truncated_normal(0.0, 1.0, -1.0, 1.0, &[5])?;
        println!(
            "- Truncated Normal (mean=0, std=1, low=-1, high=1) samples: {:?}",
            trunc_normal_samples
        );
        println!("  * Note: All samples are constrained between -1 and 1");
    }

    #[cfg(not(feature = "scirs"))]
    {
        println!("- Truncated Normal distribution: Not available (requires SciRS2 integration)");
    }

    println!("\nPart 3: Multivariate Distributions");
    println!("--------------------------------");

    // Multivariate normal distributions
    println!("\nMultivariate Normal Distributions:");

    // Setup common parameters
    #[cfg(feature = "scirs")]
    let mean = vec![0.0, 0.0];
    #[cfg(feature = "scirs")]
    let cov_data = vec![1.0, 0.7, 0.7, 1.0]; // Correlation of 0.7
    #[cfg(feature = "scirs")]
    let cov = Array::from_vec(cov_data).reshape(&[2, 2]);

    #[cfg(feature = "scirs")]
    {
        // Without rotation
        let mvn_samples = multivariate_normal_with_rotation(&mean, &cov, Some(&[3]), None)?;
        println!("- Multivariate Normal (2D, corr=0.7, no rotation) samples:");
        println!("  {:?}", mvn_samples);

        // With 45-degree rotation
        let rot_data = vec![
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2, // cos(45°), sin(45°)
            -std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2, // -sin(45°), cos(45°)
        ];
        let rotation = Array::from_vec(rot_data).reshape(&[2, 2]);

        let mvn_rot_samples =
            multivariate_normal_with_rotation(&mean, &cov, Some(&[3]), Some(&rotation))?;
        println!("- Multivariate Normal (2D, corr=0.7, 45° rotation) samples:");
        println!("  {:?}", mvn_rot_samples);
        println!("  * Note: Rotation changes the coordinate system while preserving the covariance structure");
    }

    #[cfg(not(feature = "scirs"))]
    {
        println!(
            "- Multivariate Normal with rotation: Not available (requires SciRS2 integration)"
        );
    }

    println!("\nPart 4: Statistical Applications");
    println!("------------------------------");

    #[cfg(feature = "scirs")]
    {
        println!("\nDemonstrating Hypothesis Testing with Noncentral Distributions:");

        // Generate 1000 samples from a noncentral chi-square distribution
        println!("- Generating 1000 samples from noncentral chi-square (df=10, nonc=5)...");
        let large_nc_chi2 = noncentral_chisquare(10.0, 5.0, &[1000])?;
        let data = large_nc_chi2.to_vec();

        // Calculate mean and compare to theoretical
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let expected_mean = 10.0 + 5.0; // df + nonc

        println!("  * Sample mean = {:.4}", mean);
        println!("  * Theoretical mean = {:.4}", expected_mean);
        println!(
            "  * Difference = {:.4}%",
            ((mean - expected_mean) / expected_mean * 100.0).abs()
        );

        // Count values exceeding a critical value (95th percentile of central chi-square)
        // This simulates a power analysis for a test
        let chi2_critical = 18.31; // 95th percentile of central chi-square(10)
        let count_exceeding = data.iter().filter(|&&x| x > chi2_critical).count();
        let power = count_exceeding as f64 / 1000.0;

        println!("  * Samples exceeding critical value = {}", count_exceeding);
        println!("  * Empirical power = {:.2}", power);
        println!("  * This demonstrates how noncentral distributions are used in power analysis");
    }

    println!("\nSummary of SciRS2 Integration Benefits");
    println!("------------------------------------");
    println!("1. Advanced distributions not available in standard NumRS2");
    println!("2. Special-purpose distributions for scientific computing");
    println!("3. Extended functionality like multivariate distributions with rotation");
    println!("4. Statistical applications including hypothesis testing and power analysis");

    println!("\nHow to Enable SciRS2 Integration");
    println!("------------------------------");
    println!("Add the following to your Cargo.toml:");
    println!("numrs2 = {{ version = \"0.1.1\", features = [\"scirs\"] }}");
    println!("Or when building: cargo build --features scirs");

    Ok(())
}
