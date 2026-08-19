#![allow(deprecated)]

// Example demonstrating statistical functions in NumRS2
//
// This example shows how to use various statistical functions in NumRS2,
// including descriptive statistics, correlation, histograms, and more.

use numrs2::prelude::*;
use numrs2::stats::{
    average, bincount, corrcoef, cov, digitize, histogram, histogram2d, percentile, ptp, quantile,
};

#[allow(clippy::result_large_err)]
fn main() -> numrs2::error::Result<()> {
    println!("NumRS2 Statistical Functions Example");
    println!("====================================");

    // Create sample data
    let data1 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let data2 = Array::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);

    println!("\nPart 1: Basic Descriptive Statistics");
    println!("----------------------------------");

    // Basic statistics using the Statistics trait
    println!("Data1: {}", data1);
    println!("Mean:   {}", data1.mean());
    println!("Var:    {}", data1.var());
    println!("Std:    {}", data1.std());
    println!("Min:    {}", data1.min());
    println!("Max:    {}", data1.max());

    // Range (peak-to-peak)
    let range = ptp(&data1, None)?;
    println!("Range:  {}", range);

    // Percentiles
    let percentiles = Array::from_vec(vec![0.0, 25.0, 50.0, 75.0, 100.0]);
    let values = percentile(&data1, &percentiles, Some("linear"))?;
    println!("Percentiles:");
    for (p, v) in percentiles.to_vec().iter().zip(values.to_vec().iter()) {
        println!("  {}%: {}", p, v);
    }

    println!("\nPart 2: Correlation and Covariance");
    println!("--------------------------------");

    // Covariance and correlation
    println!("Data1: {}", data1);
    println!("Data2: {}", data2);
    println!(
        "Covariance:   {}",
        cov(&data1, Some(&data2), None, None, None)?
    );
    println!("Correlation:  {}", corrcoef(&data1, Some(&data2), None)?);

    // Create positively correlated data
    let data3 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let data4 = Array::from_vec(vec![2.0, 3.5, 5.0, 6.5, 8.0]); // y = 1.5x + 0.5

    println!("\nPositively correlated data:");
    println!("Data3: {}", data3);
    println!("Data4: {}", data4);
    println!(
        "Covariance:   {}",
        cov(&data3, Some(&data4), None, None, None)?
    );
    println!("Correlation:  {}", corrcoef(&data3, Some(&data4), None)?);

    println!("\nPart 3: Histograms");
    println!("-----------------");

    // Create data for histograms
    let normal_data = create_normal_distribution_data(100, 5.0, 2.0);
    println!("Created normal distribution data with mean=5.0, std=2.0");

    // Calculate histogram
    let (hist, bin_edges) = histogram(&normal_data, 10, None, None)?;

    println!("\nHistogram (10 bins):");
    println!("Counts:    {}", hist);
    println!("Bin edges: {}", bin_edges);

    // Create 2D data for 2D histogram
    let x_data = create_normal_distribution_data(200, 0.0, 1.0);
    let y_data = create_normal_distribution_data(200, 0.0, 1.0);

    // Calculate 2D histogram
    let (hist2d, x_edges, y_edges) = histogram2d(&x_data, &y_data, 5, None, None)?;

    println!("\n2D Histogram (5x5 bins):");
    println!("2D Histogram shape: {:?}", hist2d.shape());
    println!("X bin edges: {}", x_edges);
    println!("Y bin edges: {}", y_edges);
    println!("2D Histogram:\n{}", hist2d);

    println!("\nPart 4: Binning and Counting");
    println!("---------------------------");

    // Create data for bincount
    let count_data = Array::from_vec(vec![0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]);
    println!("Count data: {}", count_data);

    // Calculate bincount
    let counts = bincount(&count_data, None, Some(5))?;
    println!("Bin counts: {}", counts);

    // Create data for digitize
    let values = Array::from_vec(vec![0.2, 1.4, 2.5, 3.7, 4.8]);
    let bins = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    println!("\nValues:    {}", values);
    println!("Bins:      {}", bins);

    // Calculate digitize
    let indices = digitize(&values, &bins, Some(false))?;
    println!("Bin indices (right=false): {}", indices);

    let indices = digitize(&values, &bins, Some(true))?;
    println!("Bin indices (right=true):  {}", indices);

    println!("\nPart 5: Weighted Statistics");
    println!("--------------------------");

    // Create data and weights
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let weights = Array::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);

    println!("Data:    {}", data);
    println!("Weights: {}", weights);

    // Calculate weighted average
    let weighted_avg = average(&data, Some(&weights), None)?;
    println!("Weighted average: {}", weighted_avg);

    // Regular average for comparison
    let avg = average(&data, None, None)?;
    println!("Regular average:  {}", avg);

    // Weighted histogram
    let (hist, bin_edges) = histogram(&data, 5, None, Some(&weights))?;
    println!("\nWeighted histogram:");
    println!("Counts:    {}", hist);
    println!("Bin edges: {}", bin_edges);

    println!("\nPart 6: Quantiles");
    println!("----------------");

    // Create some more complex data
    let big_data = create_normal_distribution_data(1000, 0.0, 1.0);

    // Define quantiles to calculate
    let q_values = Array::from_vec(vec![0.1, 0.25, 0.5, 0.75, 0.9]);

    // Calculate quantiles with different methods
    let q_linear = quantile(&big_data, &q_values, Some("linear"))?;
    let q_lower = quantile(&big_data, &q_values, Some("lower"))?;
    let q_higher = quantile(&big_data, &q_values, Some("higher"))?;
    let q_nearest = quantile(&big_data, &q_values, Some("nearest"))?;
    let q_midpoint = quantile(&big_data, &q_values, Some("midpoint"))?;

    println!("Quantiles for normal distribution (µ=0, σ=1, n=1000):");
    println!("Quantile points: {}", q_values);
    println!("Linear method:   {}", q_linear);
    println!("Lower method:    {}", q_lower);
    println!("Higher method:   {}", q_higher);
    println!("Nearest method:  {}", q_nearest);
    println!("Midpoint method: {}", q_midpoint);

    Ok(())
}

// Helper function to create normally distributed data
fn create_normal_distribution_data(n: usize, mean: f64, std: f64) -> Array<f64> {
    use numrs2::random::default_rng;

    let rng = default_rng();
    rng.normal(mean, std, &[n]).unwrap()
}
