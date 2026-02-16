//! Comprehensive Statistical Analysis Example for NumRS2
//!
//! This example demonstrates advanced statistical analysis including:
//! - Distribution fitting and parameter estimation
//! - Hypothesis testing (t-tests, chi-square, ANOVA)
//! - Statistical inference and confidence intervals
//! - Bootstrapping and resampling methods
//! - Correlation and regression analysis
//!
//! Run with: cargo run --example statistical_analysis

use numrs2::prelude::*;
use numrs2::random::default_rng;
use numrs2::stats::*;
use std::f64::consts::PI;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== NumRS2 Statistical Analysis Examples ===\n");

    // Example 1: Distribution Fitting
    example1_distribution_fitting()?;

    // Example 2: Hypothesis Testing
    example2_hypothesis_testing()?;

    // Example 3: Confidence Intervals
    example3_confidence_intervals()?;

    // Example 4: Bootstrapping
    example4_bootstrapping()?;

    // Example 5: Correlation Analysis
    example5_correlation_analysis()?;

    // Example 6: Regression Analysis
    example6_regression_analysis()?;

    // Example 7: Statistical Inference
    example7_statistical_inference()?;

    println!("\n=== All Statistical Analysis Examples Completed Successfully! ===");
    Ok(())
}

/// Example 1: Distribution Fitting
fn example1_distribution_fitting() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Distribution Fitting");
    println!("================================\n");

    let rng = default_rng();

    // 1.1 Fit Normal Distribution
    println!("1.1 Normal Distribution Fitting");

    // Generate sample data from N(5, 2²)
    let true_mean = 5.0;
    let true_std = 2.0;
    let sample_size = 1000;

    let data = rng.normal(true_mean, true_std, &[sample_size])?;

    // Estimate parameters using sample statistics
    let estimated_mean = data.mean();
    let estimated_std = data.std();

    println!("  True parameters: μ = {}, σ = {}", true_mean, true_std);
    println!(
        "  Estimated parameters: μ = {:.4}, σ = {:.4}",
        estimated_mean, estimated_std
    );
    println!("  Sample size: {}", sample_size);

    // Calculate goodness of fit using chi-square
    let (hist, bin_edges) = histogram(&data, 20, None, None)?;
    println!("  Histogram bins: {}", hist.size());
    println!("  Data range: [{:.2}, {:.2}]\n", data.min(), data.max());

    // 1.2 Fit Uniform Distribution
    println!("1.2 Uniform Distribution Fitting");

    let low = 0.0;
    let high = 10.0;
    let data = rng.uniform(low, high, &[sample_size])?;

    let estimated_min = data.min();
    let estimated_max = data.max();
    let estimated_mean = data.mean();

    println!("  True parameters: low = {}, high = {}", low, high);
    println!(
        "  Estimated range: [{:.4}, {:.4}]",
        estimated_min, estimated_max
    );
    println!(
        "  Estimated mean: {:.4} (expected: {:.1})",
        estimated_mean,
        (low + high) / 2.0
    );
    println!("  Sample size: {}\n", sample_size);

    // 1.3 Fit Exponential Distribution
    println!("1.3 Exponential Distribution Fitting");

    let rate = 0.5; // lambda = 0.5
    let data = rng.exponential(rate, &[sample_size])?;

    let estimated_mean = data.mean();
    let estimated_rate = 1.0 / estimated_mean;

    println!("  True rate (λ): {}", rate);
    println!("  Estimated rate (λ): {:.4}", estimated_rate);
    println!(
        "  True mean: {:.2}, Estimated mean: {:.4}",
        1.0 / rate,
        estimated_mean
    );
    println!("  Sample size: {}\n", sample_size);

    println!("✓ Example 1 completed\n");
    Ok(())
}

/// Example 2: Hypothesis Testing
fn example2_hypothesis_testing() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Hypothesis Testing");
    println!("==============================\n");

    let rng = default_rng();

    // 2.1 One-Sample t-test
    println!("2.1 One-Sample t-test");

    // H₀: μ = 100 vs H₁: μ ≠ 100
    let null_mean = 100.0;
    let data = rng.normal(102.0, 10.0, &[50])?; // True mean = 102

    let sample_mean = data.mean();
    let sample_std = data.std();
    let n = data.size() as f64;

    // Calculate t-statistic
    let t_stat = (sample_mean - null_mean) / (sample_std / n.sqrt());
    let degrees_of_freedom = n - 1.0;

    println!("  Null hypothesis: μ = {}", null_mean);
    println!("  Sample mean: {:.4}", sample_mean);
    println!("  Sample std: {:.4}", sample_std);
    println!("  Sample size: {}", n);
    println!("  t-statistic: {:.4}", t_stat);
    println!("  Degrees of freedom: {}", degrees_of_freedom);

    // Critical value for α = 0.05, two-tailed (approximate)
    let critical_value = 2.01; // t(49, 0.025) ≈ 2.01
    let reject_null = t_stat.abs() > critical_value;

    println!("  Critical value (α=0.05): ±{}", critical_value);
    println!(
        "  Decision: {} null hypothesis",
        if reject_null {
            "Reject"
        } else {
            "Fail to reject"
        }
    );
    println!();

    // 2.2 Two-Sample t-test
    println!("2.2 Two-Sample t-test");

    // H₀: μ₁ = μ₂ vs H₁: μ₁ ≠ μ₂
    let group1 = rng.normal(100.0, 10.0, &[30])?;
    let group2 = rng.normal(105.0, 10.0, &[30])?;

    let mean1 = group1.mean();
    let mean2 = group2.mean();
    let std1 = group1.std();
    let std2 = group2.std();
    let n1 = group1.size() as f64;
    let n2 = group2.size() as f64;

    // Pooled standard deviation
    let sp = (((n1 - 1.0) * std1 * std1 + (n2 - 1.0) * std2 * std2) / (n1 + n2 - 2.0)).sqrt();

    // t-statistic for equal variance
    let t_stat = (mean1 - mean2) / (sp * (1.0 / n1 + 1.0 / n2).sqrt());

    println!("  Group 1: n={}, mean={:.4}, std={:.4}", n1, mean1, std1);
    println!("  Group 2: n={}, mean={:.4}, std={:.4}", n2, mean2, std2);
    println!("  Mean difference: {:.4}", mean1 - mean2);
    println!("  Pooled std: {:.4}", sp);
    println!("  t-statistic: {:.4}", t_stat);

    let critical_value = 2.00; // t(58, 0.025) ≈ 2.00
    let reject_null = t_stat.abs() > critical_value;

    println!("  Critical value (α=0.05): ±{}", critical_value);
    println!(
        "  Decision: {} null hypothesis",
        if reject_null {
            "Reject"
        } else {
            "Fail to reject"
        }
    );
    println!();

    // 2.3 Chi-Square Goodness of Fit Test
    println!("2.3 Chi-Square Goodness of Fit Test");

    // Test if a die is fair
    let observed = Array::from_vec(vec![48.0, 52.0, 45.0, 53.0, 50.0, 52.0]);
    let total = observed.sum();
    let expected = Array::from_vec(vec![total / 6.0; 6]);

    // Calculate chi-square statistic
    let mut chi_square: f64 = 0.0;
    for i in 0..6 {
        let obs: f64 = observed.get(&[i])?;
        let exp: f64 = expected.get(&[i])?;
        chi_square += (obs - exp).powi(2_i32) / exp;
    }

    println!("  Observed frequencies: {:?}", observed.to_vec());
    println!("  Expected frequencies: {:?}", expected.to_vec());
    println!("  Chi-square statistic: {:.4}", chi_square);
    println!("  Degrees of freedom: 5");

    let critical_value = 11.07; // χ²(5, 0.05) ≈ 11.07
    let reject_null = chi_square > critical_value;

    println!("  Critical value (α=0.05): {}", critical_value);
    println!(
        "  Decision: {} null hypothesis (die is fair)",
        if reject_null {
            "Reject"
        } else {
            "Fail to reject"
        }
    );
    println!();

    println!("✓ Example 2 completed\n");
    Ok(())
}

/// Example 3: Confidence Intervals
fn example3_confidence_intervals() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Confidence Intervals");
    println!("================================\n");

    let rng = default_rng();

    // 3.1 Confidence Interval for Mean (Normal Distribution)
    println!("3.1 Confidence Interval for Mean");

    let data = rng.normal(50.0, 10.0, &[100])?;
    let sample_mean = data.mean();
    let sample_std = data.std();
    let n = data.size() as f64;

    // 95% CI using t-distribution
    let t_critical = 1.984; // t(99, 0.025) ≈ 1.984
    let margin_of_error = t_critical * sample_std / n.sqrt();
    let ci_lower = sample_mean - margin_of_error;
    let ci_upper = sample_mean + margin_of_error;

    println!("  Sample mean: {:.4}", sample_mean);
    println!("  Sample std: {:.4}", sample_std);
    println!("  Sample size: {}", n);
    println!("  95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);
    println!("  True mean: 50.0");
    println!(
        "  True mean in CI: {}",
        ci_lower <= 50.0 && 50.0 <= ci_upper
    );
    println!();

    // 3.2 Confidence Interval for Proportion
    println!("3.2 Confidence Interval for Proportion");

    let successes: f64 = 65.0;
    let n: f64 = 100.0;
    let p_hat: f64 = successes / n;

    // 95% CI using normal approximation
    let z_critical: f64 = 1.96; // z(0.025) ≈ 1.96
    let variance: f64 = p_hat * (1.0 - p_hat) / n;
    let se: f64 = variance.sqrt();
    let ci_lower = p_hat - z_critical * se;
    let ci_upper = p_hat + z_critical * se;

    println!("  Sample proportion: {:.4}", p_hat);
    println!("  Sample size: {}", n);
    println!("  Standard error: {:.4}", se);
    println!("  95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);
    println!();

    // 3.3 Confidence Interval for Variance
    println!("3.3 Confidence Interval for Variance");

    let data = rng.normal(0.0, 5.0, &[50])?;
    let sample_var = data.var();
    let n = data.size() as f64;

    // 95% CI using chi-square distribution
    let chi2_lower = 32.36; // χ²(49, 0.975) ≈ 32.36
    let chi2_upper = 70.22; // χ²(49, 0.025) ≈ 70.22

    let ci_lower = (n - 1.0) * sample_var / chi2_upper;
    let ci_upper = (n - 1.0) * sample_var / chi2_lower;

    println!("  Sample variance: {:.4}", sample_var);
    println!("  Sample size: {}", n);
    println!("  95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);
    println!("  True variance: 25.0");
    println!(
        "  True variance in CI: {}",
        ci_lower <= 25.0 && 25.0 <= ci_upper
    );
    println!();

    println!("✓ Example 3 completed\n");
    Ok(())
}

/// Example 4: Bootstrapping
fn example4_bootstrapping() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 4: Bootstrapping");
    println!("========================\n");

    let rng = default_rng();

    // Original sample
    let original_sample = rng.normal(100.0, 15.0, &[30])?;
    let original_mean = original_sample.mean();

    println!("4.1 Bootstrap Confidence Interval for Mean");
    println!("  Original sample size: {}", original_sample.size());
    println!("  Original sample mean: {:.4}", original_mean);

    // Bootstrap resampling
    let n_bootstrap = 1000;
    let n = original_sample.size();
    let mut bootstrap_means = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let indices = rng.integers(0, n as i64, &[n])?;
        let mut resample = Vec::with_capacity(n);

        for i in 0..n {
            let idx = indices.get(&[i])? as usize;
            resample.push(original_sample.get(&[idx])?);
        }

        let resample_array = Array::from_vec(resample);
        bootstrap_means.push(resample_array.mean());
    }

    // Sort bootstrap means for percentile method
    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // 95% CI using percentile method
    let lower_idx = (n_bootstrap as f64 * 0.025) as usize;
    let upper_idx = (n_bootstrap as f64 * 0.975) as usize;
    let ci_lower = bootstrap_means[lower_idx];
    let ci_upper = bootstrap_means[upper_idx];

    println!("  Bootstrap samples: {}", n_bootstrap);
    println!("  Bootstrap 95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);

    // Bootstrap standard error
    let bootstrap_mean_array = Array::from_vec(bootstrap_means.clone());
    let bootstrap_se = bootstrap_mean_array.std();
    println!("  Bootstrap standard error: {:.4}", bootstrap_se);
    println!();

    // 4.2 Bootstrap for Correlation Coefficient
    println!("4.2 Bootstrap Confidence Interval for Correlation");

    let x = rng.normal(0.0, 1.0, &[50])?;
    let y_data: Vec<f64> = (0..50)
        .map(|i| {
            let xi = x.get(&[i]).unwrap();
            0.7 * xi + rng.normal(0.0, 0.5, &[1]).unwrap().get(&[0]).unwrap()
        })
        .collect();
    let y = Array::from_vec(y_data);

    // Original correlation
    let original_corr = corrcoef(&x, Some(&y), None)?;
    println!("  Original correlation: {:.4}", original_corr);

    let mut bootstrap_corrs = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let indices = rng.integers(0, 50i64, &[50])?;
        let mut resample_x = Vec::with_capacity(50);
        let mut resample_y = Vec::with_capacity(50);

        for i in 0..50 {
            let idx = indices.get(&[i])? as usize;
            resample_x.push(x.get(&[idx])?);
            resample_y.push(y.get(&[idx])?);
        }

        let x_array = Array::from_vec(resample_x);
        let y_array = Array::from_vec(resample_y);
        let corr = corrcoef(&x_array, Some(&y_array), None)?;
        // Extract scalar value from SpecialArray
        let corr_value: f64 = corr.get(&[])?;
        bootstrap_corrs.push(corr_value);
    }

    bootstrap_corrs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let ci_lower = bootstrap_corrs[lower_idx];
    let ci_upper = bootstrap_corrs[upper_idx];

    println!("  Bootstrap 95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);
    println!();

    println!("✓ Example 4 completed\n");
    Ok(())
}

/// Example 5: Correlation Analysis
fn example5_correlation_analysis() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 5: Correlation Analysis");
    println!("================================\n");

    let rng = default_rng();

    // 5.1 Pearson Correlation
    println!("5.1 Pearson Correlation (Linear Relationship)");

    let x = Array::from_vec((0..100).map(|i| i as f64).collect());
    let y_data: Vec<f64> = (0..100)
        .map(|i| {
            let xi = x.get(&[i]).unwrap();
            2.0 * xi + 5.0 + rng.normal(0.0, 10.0, &[1]).unwrap().get(&[0]).unwrap()
        })
        .collect();
    let y = Array::from_vec(y_data);

    let correlation = corrcoef(&x, Some(&y), None)?;
    let corr_abs = correlation.abs();
    // Extract scalar values for comparison
    let corr_value: f64 = correlation.get(&[])?;
    let corr_abs_value: f64 = corr_abs.get(&[])?;
    println!("  Relationship: y ≈ 2x + 5 + noise");
    println!("  Pearson correlation: {:.4}", corr_value);
    println!(
        "  Interpretation: {} linear relationship",
        if corr_abs_value > 0.7 {
            "Strong"
        } else if corr_abs_value > 0.4 {
            "Moderate"
        } else {
            "Weak"
        }
    );
    println!();

    // 5.2 Covariance
    println!("5.2 Covariance Matrix");

    let data1 = rng.normal(50.0, 10.0, &[100])?;
    let data2_vec: Vec<f64> = (0..100)
        .map(|i| {
            data1.get(&[i]).unwrap() * 0.8 + rng.normal(0.0, 5.0, &[1]).unwrap().get(&[0]).unwrap()
        })
        .collect();
    let data2 = Array::from_vec(data2_vec);

    let covariance = cov(&data1, Some(&data2), None, None, None)?;
    let var1 = data1.var();
    let var2 = data2.var();

    println!("  Variable 1 - mean: {:.2}, var: {:.2}", data1.mean(), var1);
    println!("  Variable 2 - mean: {:.2}, var: {:.2}", data2.mean(), var2);
    println!("  Covariance: {:.4}", covariance);
    println!(
        "  Correlation (from cov): {:.4}",
        covariance / (var1.sqrt() * var2.sqrt())
    );
    println!();

    // 5.3 Multiple Variables Correlation
    println!("5.3 Pairwise Correlations");

    let var1 = rng.normal(0.0, 1.0, &[50])?;
    let var2_vec: Vec<f64> = (0..50)
        .map(|i| {
            0.9 * var1.get(&[i]).unwrap() + rng.normal(0.0, 0.3, &[1]).unwrap().get(&[0]).unwrap()
        })
        .collect();
    let var2 = Array::from_vec(var2_vec);

    let var3 = rng.normal(0.0, 1.0, &[50])?; // Independent

    let corr12 = corrcoef(&var1, Some(&var2), None)?;
    let corr13 = corrcoef(&var1, Some(&var3), None)?;
    let corr23 = corrcoef(&var2, Some(&var3), None)?;

    println!("  Correlation matrix:");
    println!("         Var1   Var2   Var3");
    println!("  Var1  1.0000 {:.4} {:.4}", corr12, corr13);
    println!("  Var2  {:.4} 1.0000 {:.4}", corr12, corr23);
    println!("  Var3  {:.4} {:.4} 1.0000", corr13, corr23);
    println!();

    println!("✓ Example 5 completed\n");
    Ok(())
}

/// Example 6: Regression Analysis
fn example6_regression_analysis() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 6: Regression Analysis");
    println!("===============================\n");

    let rng = default_rng();

    // 6.1 Simple Linear Regression
    println!("6.1 Simple Linear Regression");

    let x = Array::from_vec((1..=100).map(|i| i as f64).collect());
    let true_slope = 2.5;
    let true_intercept = 10.0;

    let y_data: Vec<f64> = (0..100)
        .map(|i| {
            let xi = x.get(&[i]).unwrap();
            true_slope * xi
                + true_intercept
                + rng.normal(0.0, 5.0, &[1]).unwrap().get(&[0]).unwrap()
        })
        .collect();
    let y = Array::from_vec(y_data);

    // Calculate regression coefficients
    let n = x.size() as f64;
    let mean_x = x.mean();
    let mean_y = y.mean();

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..100 {
        let xi: f64 = x.get(&[i])?;
        let yi: f64 = y.get(&[i])?;
        numerator += (xi - mean_x) * (yi - mean_y);
        denominator += (xi - mean_x).powi(2_i32);
    }

    let slope = numerator / denominator;
    let intercept = mean_y - slope * mean_x;

    println!(
        "  True model: y = {:.1}x + {:.1}",
        true_slope, true_intercept
    );
    println!("  Fitted model: y = {:.4}x + {:.4}", slope, intercept);

    // Calculate R²
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for i in 0..100 {
        let xi: f64 = x.get(&[i])?;
        let yi: f64 = y.get(&[i])?;
        let y_pred: f64 = slope * xi + intercept;

        ss_tot += (yi - mean_y).powi(2_i32);
        ss_res += (yi - y_pred).powi(2_i32);
    }

    let r_squared = 1.0 - ss_res / ss_tot;
    println!("  R² (coefficient of determination): {:.6}", r_squared);

    // Standard error of estimate
    let se = (ss_res / (n - 2.0)).sqrt();
    println!("  Standard error of estimate: {:.4}", se);
    println!();

    // 6.2 Residual Analysis
    println!("6.2 Residual Analysis");

    let mut residuals = Vec::with_capacity(100);
    for i in 0..100 {
        let xi = x.get(&[i])?;
        let yi = y.get(&[i])?;
        let y_pred = slope * xi + intercept;
        residuals.push(yi - y_pred);
    }

    let residuals_array = Array::from_vec(residuals);
    println!("  Residual mean: {:.6}", residuals_array.mean());
    println!("  Residual std: {:.4}", residuals_array.std());
    println!("  Min residual: {:.4}", residuals_array.min());
    println!("  Max residual: {:.4}", residuals_array.max());
    println!();

    println!("✓ Example 6 completed\n");
    Ok(())
}

/// Example 7: Statistical Inference
fn example7_statistical_inference() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 7: Statistical Inference");
    println!("=================================\n");

    let rng = default_rng();

    // 7.1 Point Estimation
    println!("7.1 Point Estimation (Method of Moments)");

    // Generate data from known distribution
    let true_mean = 75.0;
    let true_var: f64 = 100.0;
    let data = rng.normal(true_mean, true_var.sqrt(), &[200])?;

    // Method of moments estimators
    let mom_mean = data.mean();
    let mom_var = data.var();

    println!("  True parameters: μ = {}, σ² = {}", true_mean, true_var);
    println!("  MOM estimates: μ̂ = {:.4}, σ̂² = {:.4}", mom_mean, mom_var);
    println!();

    // 7.2 Quantile Estimation
    println!("7.2 Quantile Estimation");

    let quantiles = Array::from_vec(vec![0.25, 0.50, 0.75, 0.90, 0.95, 0.99]);
    let estimated_quantiles = quantile(&data, &quantiles, Some("linear"))?;

    println!("  Sample quantiles:");
    for i in 0..quantiles.size() {
        let q = quantiles.get(&[i])?;
        let val = estimated_quantiles.get(&[i])?;
        println!("    Q({:.0}%): {:.4}", q * 100.0, val);
    }
    println!();

    // 7.3 Order Statistics
    println!("7.3 Order Statistics");

    let small_sample = rng.normal(50.0, 10.0, &[20])?;
    let min_val = small_sample.min();
    let max_val = small_sample.max();
    let median_val = quantile(&small_sample, &Array::from_vec(vec![0.5]), Some("linear"))?;

    println!("  Sample size: {}", small_sample.size());
    println!("  Minimum (X₍₁₎): {:.4}", min_val);
    println!("  Maximum (X₍ₙ₎): {:.4}", max_val);
    println!("  Range: {:.4}", max_val - min_val);
    println!("  Median: {:.4}", median_val.get(&[0])?);
    println!();

    println!("✓ Example 7 completed\n");
    Ok(())
}
