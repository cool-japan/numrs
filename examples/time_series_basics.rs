//! Time Series Analysis Basics for NumRS2
//!
//! This example demonstrates fundamental time series analysis including:
//! - Moving averages and smoothing techniques
//! - Autocorrelation and partial autocorrelation
//! - Trend analysis and decomposition
//! - Basic forecasting methods
//! - Seasonal decomposition
//! - Differencing and stationarity
//!
//! Run with: cargo run --example time_series_basics

use numrs2::prelude::*;
use numrs2::random::default_rng;
use std::f64::consts::PI;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== NumRS2 Time Series Analysis Examples ===\n");

    // Example 1: Moving Averages
    example1_moving_averages()?;

    // Example 2: Exponential Smoothing
    example2_exponential_smoothing()?;

    // Example 3: Autocorrelation Analysis
    example3_autocorrelation()?;

    // Example 4: Trend Analysis
    example4_trend_analysis()?;

    // Example 5: Seasonal Decomposition
    example5_seasonal_decomposition()?;

    // Example 6: Differencing for Stationarity
    example6_differencing()?;

    // Example 7: Basic Forecasting
    example7_basic_forecasting()?;

    println!("\n=== All Time Series Examples Completed Successfully! ===");
    Ok(())
}

/// Example 1: Moving Averages
fn example1_moving_averages() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 1: Moving Averages");
    println!("==========================\n");

    let rng = default_rng();

    // Generate time series with trend and noise
    let n = 100;
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64;
        let trend = 0.5 * t;
        let noise = rng.normal(0.0, 5.0, &[1])?.get(&[0])?;
        data.push(trend + noise);
    }

    let series = Array::from_vec(data);

    println!("1.1 Simple Moving Average (SMA)");

    // Calculate 5-period SMA
    let window = 5;
    let mut sma = Vec::with_capacity(n - window + 1);

    for i in 0..=(n - window) {
        let mut sum = 0.0;
        for j in 0..window {
            sum += series.get(&[i + j])?;
        }
        sma.push(sum / window as f64);
    }

    let sma_array = Array::from_vec(sma);

    println!("  Original series length: {}", n);
    println!("  Window size: {}", window);
    println!("  SMA length: {}", sma_array.size());
    println!("  First 10 values:");
    for i in 0..10.min(sma_array.size()) {
        println!("    SMA[{}]: {:.4}", i, sma_array.get(&[i])?);
    }
    println!();

    // Calculate 10-period SMA
    println!("1.2 Different Window Sizes");

    let windows = vec![3, 5, 10, 20];
    for &w in &windows {
        let mut moving_avg = Vec::with_capacity(n - w + 1);
        for i in 0..=(n - w) {
            let mut sum = 0.0;
            for j in 0..w {
                sum += series.get(&[i + j])?;
            }
            moving_avg.push(sum / w as f64);
        }

        let ma_array = Array::from_vec(moving_avg);
        println!(
            "  Window {}: mean={:.4}, std={:.4}",
            w,
            ma_array.mean(),
            ma_array.std()
        );
    }
    println!();

    // Weighted Moving Average
    println!("1.3 Weighted Moving Average (WMA)");

    let window = 5;
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Linear weights
    let weight_sum: f64 = weights.iter().sum();

    let mut wma = Vec::with_capacity(n - window + 1);
    for i in 0..=(n - window) {
        let mut weighted_sum = 0.0;
        for (j, &weight) in weights.iter().enumerate() {
            weighted_sum += series.get(&[i + j])? * weight;
        }
        wma.push(weighted_sum / weight_sum);
    }

    let wma_array = Array::from_vec(wma);
    println!("  Weights: {:?}", weights);
    println!("  First 5 WMA values:");
    for i in 0..5.min(wma_array.size()) {
        println!("    WMA[{}]: {:.4}", i, wma_array.get(&[i])?);
    }
    println!();

    println!("✓ Example 1 completed\n");
    Ok(())
}

/// Example 2: Exponential Smoothing
fn example2_exponential_smoothing() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 2: Exponential Smoothing");
    println!("=================================\n");

    let rng = default_rng();

    // Generate time series
    let n = 100;
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64;
        let level = 50.0;
        let trend = 0.3 * t;
        let noise = rng.normal(0.0, 3.0, &[1])?.get(&[0])?;
        data.push(level + trend + noise);
    }

    let series = Array::from_vec(data);

    // Simple Exponential Smoothing (SES)
    println!("2.1 Simple Exponential Smoothing (SES)");

    let alpha = 0.3; // Smoothing parameter
    let mut ses = Vec::with_capacity(n);
    ses.push(series.get(&[0])?); // Initialize with first observation

    for i in 1..n {
        let smoothed = alpha * series.get(&[i])? + (1.0 - alpha) * ses[i - 1];
        ses.push(smoothed);
    }

    let ses_array = Array::from_vec(ses);

    println!("  Alpha (smoothing parameter): {}", alpha);
    println!("  First 10 smoothed values:");
    for i in 0..10 {
        println!(
            "    t={}: original={:.4}, smoothed={:.4}",
            i,
            series.get(&[i])?,
            ses_array.get(&[i])?
        );
    }
    println!();

    // Double Exponential Smoothing (Holt's method)
    println!("2.2 Double Exponential Smoothing (Holt's Linear Trend)");

    let alpha = 0.3; // Level smoothing
    let beta = 0.1; // Trend smoothing

    let mut level = Vec::with_capacity(n);
    let mut trend = Vec::with_capacity(n);
    let mut forecast = Vec::with_capacity(n);

    // Initialize
    level.push(series.get(&[0])?);
    trend.push(series.get(&[1])? - series.get(&[0])?);
    forecast.push(level[0] + trend[0]);

    for i in 1..n {
        let obs = series.get(&[i])?;

        let new_level = alpha * obs + (1.0 - alpha) * (level[i - 1] + trend[i - 1]);
        let new_trend = beta * (new_level - level[i - 1]) + (1.0 - beta) * trend[i - 1];

        level.push(new_level);
        trend.push(new_trend);
        forecast.push(new_level + new_trend);
    }

    println!("  Alpha (level): {}", alpha);
    println!("  Beta (trend): {}", beta);
    println!("  Last 5 forecasts:");
    for i in (n - 5)..n {
        println!(
            "    t={}: level={:.4}, trend={:.4}, forecast={:.4}",
            i, level[i], trend[i], forecast[i]
        );
    }
    println!();

    // Calculate forecast accuracy
    let mut mse = 0.0;
    let mut mae = 0.0;

    for i in 1..n {
        let error = series.get(&[i])? - forecast[i - 1];
        mse += error * error;
        mae += error.abs();
    }

    mse /= (n - 1) as f64;
    mae /= (n - 1) as f64;

    println!("  Forecast Accuracy:");
    println!("    MSE (Mean Squared Error): {:.4}", mse);
    println!("    RMSE (Root Mean Squared Error): {:.4}", mse.sqrt());
    println!("    MAE (Mean Absolute Error): {:.4}", mae);
    println!();

    println!("✓ Example 2 completed\n");
    Ok(())
}

/// Example 3: Autocorrelation Analysis
fn example3_autocorrelation() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 3: Autocorrelation Analysis");
    println!("====================================\n");

    let rng = default_rng();

    // Generate AR(1) process: X_t = 0.7 * X_{t-1} + noise
    let n = 200;
    let phi = 0.7;
    let mut ar1_data = Vec::with_capacity(n);
    ar1_data.push(rng.normal(0.0, 1.0, &[1])?.get(&[0])?);

    for i in 1..n {
        let prev = ar1_data[i - 1];
        let noise = rng.normal(0.0, 1.0, &[1])?.get(&[0])?;
        ar1_data.push(phi * prev + noise);
    }

    let series = Array::from_vec(ar1_data);

    println!("3.1 Autocorrelation Function (ACF)");
    println!("  Generated AR(1) process with φ = {}", phi);

    // Calculate ACF for lags 1-10
    let max_lag = 10;
    let mean = series.mean();
    let var = series.var();

    println!("\n  Autocorrelations:");
    for lag in 1..=max_lag {
        let mut sum = 0.0;
        let count = n - lag;

        for i in 0..count {
            let x_t = series.get(&[i])? - mean;
            let x_t_lag = series.get(&[i + lag])? - mean;
            sum += x_t * x_t_lag;
        }

        let acf = sum / (count as f64 * var);
        let theoretical_acf = phi.powi(lag as i32);

        // Simple significance test (approximate 95% CI)
        let se = 1.0 / (n as f64).sqrt();
        let significant = acf.abs() > 1.96 * se;

        println!(
            "    Lag {:<2}: {:.4} (theoretical: {:.4}) {}",
            lag,
            acf,
            theoretical_acf,
            if significant { "*" } else { " " }
        );
    }
    println!("  * indicates significance at 95% level");
    println!();

    // Ljung-Box test statistic
    println!("3.2 Ljung-Box Test for Serial Correlation");

    let mut lb_stat = 0.0;
    let m = 10; // Number of lags to test

    for lag in 1..=m {
        let mut sum = 0.0;
        let count = n - lag;

        for i in 0..count {
            let x_t = series.get(&[i])? - mean;
            let x_t_lag = series.get(&[i + lag])? - mean;
            sum += x_t * x_t_lag;
        }

        let acf = sum / (count as f64 * var);
        lb_stat += acf * acf / (n - lag) as f64;
    }

    lb_stat *= n as f64 * (n + 2) as f64;

    println!("  Number of lags tested: {}", m);
    println!("  Ljung-Box Q statistic: {:.4}", lb_stat);
    println!("  Critical value (χ²({}, 0.05)): ~18.3", m);
    println!(
        "  Result: {} evidence of serial correlation",
        if lb_stat > 18.3 {
            "Significant"
        } else {
            "No significant"
        }
    );
    println!();

    println!("✓ Example 3 completed\n");
    Ok(())
}

/// Example 4: Trend Analysis
fn example4_trend_analysis() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 4: Trend Analysis");
    println!("=========================\n");

    // Generate time series with linear trend
    let n = 100;
    let mut data = Vec::with_capacity(n);

    let rng = default_rng();
    let slope = 2.5;
    let intercept = 100.0;

    for i in 0..n {
        let t = i as f64;
        let trend = slope * t + intercept;
        let noise = rng.normal(0.0, 10.0, &[1])?.get(&[0])?;
        data.push(trend + noise);
    }

    let series = Array::from_vec(data);

    println!("4.1 Linear Trend Estimation (OLS)");

    // Calculate trend using ordinary least squares
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let t_array = Array::from_vec(t);

    let mean_t = t_array.mean();
    let mean_y = series.mean();

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let ti = t_array.get(&[i])?;
        let yi = series.get(&[i])?;
        numerator += (ti - mean_t) * (yi - mean_y);
        denominator += (ti - mean_t).powi(2);
    }

    let estimated_slope = numerator / denominator;
    let estimated_intercept = mean_y - estimated_slope * mean_t;

    println!("  True trend: y = {:.1}t + {:.1}", slope, intercept);
    println!(
        "  Estimated trend: y = {:.4}t + {:.4}",
        estimated_slope, estimated_intercept
    );

    // Calculate R²
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for i in 0..n {
        let ti = t_array.get(&[i])?;
        let yi = series.get(&[i])?;
        let y_pred = estimated_slope * ti + estimated_intercept;

        ss_tot += (yi - mean_y).powi(2);
        ss_res += (yi - y_pred).powi(2);
    }

    let r_squared = 1.0 - ss_res / ss_tot;
    println!("  R²: {:.6}", r_squared);
    println!();

    // Detrend the series
    println!("4.2 Detrending");

    let mut detrended = Vec::with_capacity(n);
    for i in 0..n {
        let ti = t_array.get(&[i])?;
        let yi = series.get(&[i])?;
        let trend_component = estimated_slope * ti + estimated_intercept;
        detrended.push(yi - trend_component);
    }

    let detrended_array = Array::from_vec(detrended);

    println!(
        "  Original series - mean: {:.4}, std: {:.4}",
        series.mean(),
        series.std()
    );
    println!(
        "  Detrended series - mean: {:.4}, std: {:.4}",
        detrended_array.mean(),
        detrended_array.std()
    );
    println!("  Note: Detrended series should have mean ≈ 0");
    println!();

    println!("✓ Example 4 completed\n");
    Ok(())
}

/// Example 5: Seasonal Decomposition
fn example5_seasonal_decomposition() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 5: Seasonal Decomposition");
    println!("==================================\n");

    // Generate time series with trend, seasonality, and noise
    let n = 120; // 10 years of monthly data
    let period = 12; // Monthly seasonality
    let mut data = Vec::with_capacity(n);

    let rng = default_rng();

    for i in 0..n {
        let t = i as f64;

        // Trend component
        let trend = 0.5 * t + 100.0;

        // Seasonal component (amplitude increases with time)
        let seasonal = 10.0 * (2.0 * PI * t / period as f64).sin();

        // Noise
        let noise = rng.normal(0.0, 3.0, &[1])?.get(&[0])?;

        data.push(trend + seasonal + noise);
    }

    let series = Array::from_vec(data);

    println!("5.1 Additive Decomposition (Y = T + S + E)");
    println!("  Series length: {} observations", n);
    println!("  Period: {} months", period);
    println!();

    // Extract trend using centered moving average
    println!("  Extracting trend component...");
    let window = period;
    let mut trend = vec![f64::NAN; window / 2];

    for i in (window / 2)..(n - window / 2) {
        let mut sum = 0.0;
        for j in 0..window {
            sum += series.get(&[i - window / 2 + j])?;
        }
        trend.push(sum / window as f64);
    }

    // Pad the end
    for _ in 0..(window / 2) {
        trend.push(f64::NAN);
    }

    // Calculate detrended series
    let mut detrended = Vec::with_capacity(n);
    for (i, &trend_val) in trend.iter().enumerate() {
        let value = series.get(&[i])?;
        if trend_val.is_finite() {
            detrended.push(value - trend_val);
        } else {
            detrended.push(f64::NAN);
        }
    }

    // Extract seasonal component by averaging across periods
    println!("  Extracting seasonal component...");
    let mut seasonal = vec![0.0; period];
    let mut seasonal_counts = vec![0; period];

    for (i, &detrend_val) in detrended.iter().enumerate() {
        if detrend_val.is_finite() {
            let period_idx = i % period;
            seasonal[period_idx] += detrend_val;
            seasonal_counts[period_idx] += 1;
        }
    }

    for i in 0..period {
        if seasonal_counts[i] > 0 {
            seasonal[i] /= seasonal_counts[i] as f64;
        }
    }

    // Center seasonal component (mean = 0)
    let seasonal_mean: f64 = seasonal.iter().sum::<f64>() / period as f64;
    for val in &mut seasonal {
        *val -= seasonal_mean;
    }

    println!("  Seasonal components:");
    for (i, &seasonal_val) in seasonal.iter().enumerate() {
        println!("    Month {:<2}: {:.4}", i + 1, seasonal_val);
    }
    println!();

    // Calculate residuals
    let mut residuals = Vec::with_capacity(n);
    for (i, &trend_val) in trend.iter().enumerate() {
        let value = series.get(&[i])?;
        let period_idx = i % period;

        if trend_val.is_finite() {
            let residual = value - trend_val - seasonal[period_idx];
            residuals.push(residual);
        } else {
            residuals.push(f64::NAN);
        }
    }

    let residuals_array: Vec<f64> = residuals
        .iter()
        .filter(|&&x| x.is_finite())
        .copied()
        .collect();
    let residuals_finite = Array::from_vec(residuals_array);

    println!("  Component statistics:");
    println!(
        "    Original series - mean: {:.4}, std: {:.4}",
        series.mean(),
        series.std()
    );
    println!(
        "    Residuals - mean: {:.4}, std: {:.4}",
        residuals_finite.mean(),
        residuals_finite.std()
    );
    println!();

    println!("✓ Example 5 completed\n");
    Ok(())
}

/// Example 6: Differencing for Stationarity
fn example6_differencing() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 6: Differencing for Stationarity");
    println!("=========================================\n");

    let rng = default_rng();

    // Generate non-stationary series (random walk)
    let n = 100;
    let mut random_walk = Vec::with_capacity(n);
    random_walk.push(0.0);

    for i in 1..n {
        let prev = random_walk[i - 1];
        let change = rng.normal(0.0, 1.0, &[1])?.get(&[0])?;
        random_walk.push(prev + change);
    }

    let series = Array::from_vec(random_walk);

    println!("6.1 First-Order Differencing");
    println!("  Original series (random walk):");
    println!("    Mean: {:.4}", series.mean());
    println!("    Std: {:.4}", series.std());
    println!("    First 5 values: ");
    for i in 0..5 {
        println!("      Y[{}] = {:.4}", i, series.get(&[i])?);
    }
    println!();

    // First difference: ΔY_t = Y_t - Y_{t-1}
    let mut diff1 = Vec::with_capacity(n - 1);
    for i in 1..n {
        diff1.push(series.get(&[i])? - series.get(&[i - 1])?);
    }

    let diff1_array = Array::from_vec(diff1);

    println!("  First-differenced series:");
    println!("    Mean: {:.4}", diff1_array.mean());
    println!("    Std: {:.4}", diff1_array.std());
    println!("    First 5 differences:");
    for i in 0..5 {
        println!("      ΔY[{}] = {:.4}", i + 1, diff1_array.get(&[i])?);
    }
    println!("    Note: First differencing should make random walk stationary");
    println!();

    // Second-order differencing
    println!("6.2 Second-Order Differencing");

    // Generate series with quadratic trend
    let mut quadratic_data = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        let value = 0.01 * t * t + 0.5 * t + 100.0 + rng.normal(0.0, 2.0, &[1])?.get(&[0])?;
        quadratic_data.push(value);
    }

    let quadratic_series = Array::from_vec(quadratic_data);

    // First difference
    let mut diff1 = Vec::with_capacity(n - 1);
    for i in 1..n {
        diff1.push(quadratic_series.get(&[i])? - quadratic_series.get(&[i - 1])?);
    }

    // Second difference
    let mut diff2 = Vec::with_capacity(n - 2);
    for i in 1..(n - 1) {
        diff2.push(diff1[i] - diff1[i - 1]);
    }

    let diff2_array = Array::from_vec(diff2);

    println!("  Original series (quadratic trend):");
    println!(
        "    Mean: {:.4}, Std: {:.4}",
        quadratic_series.mean(),
        quadratic_series.std()
    );

    let diff1_array = Array::from_vec(diff1);
    println!("  First difference:");
    println!(
        "    Mean: {:.4}, Std: {:.4}",
        diff1_array.mean(),
        diff1_array.std()
    );

    println!("  Second difference:");
    println!(
        "    Mean: {:.4}, Std: {:.4}",
        diff2_array.mean(),
        diff2_array.std()
    );
    println!("    Note: Second differencing removes quadratic trend");
    println!();

    println!("✓ Example 6 completed\n");
    Ok(())
}

/// Example 7: Basic Forecasting
fn example7_basic_forecasting() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("Example 7: Basic Forecasting Methods");
    println!("=====================================\n");

    let rng = default_rng();

    // Generate historical data
    let n = 50;
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64;
        let value = 100.0 + 2.0 * t + rng.normal(0.0, 5.0, &[1])?.get(&[0])?;
        data.push(value);
    }

    let series = Array::from_vec(data);

    // 7.1 Naive Forecast (last value)
    println!("7.1 Naive Forecast");

    let last_value = series.get(&[n - 1])?;
    let horizon = 10; // Forecast 10 periods ahead

    println!("  Last observed value: {:.4}", last_value);
    println!("  Forecasting {} periods ahead", horizon);
    println!("  Naive forecast (all periods): {:.4}", last_value);
    println!();

    // 7.2 Average Method
    println!("7.2 Average Method (Mean Forecast)");

    let mean_forecast = series.mean();
    println!("  Historical mean: {:.4}", mean_forecast);
    println!("  Forecast (all periods): {:.4}", mean_forecast);
    println!();

    // 7.3 Drift Method
    println!("7.3 Drift Method");

    let first_value = series.get(&[0])?;
    let drift = (last_value - first_value) / (n - 1) as f64;

    println!("  First value: {:.4}", first_value);
    println!("  Last value: {:.4}", last_value);
    println!("  Estimated drift: {:.4} per period", drift);
    println!("  Forecasts:");

    for h in 1..=horizon {
        let forecast = last_value + h as f64 * drift;
        println!("    h={}: {:.4}", h, forecast);
    }
    println!();

    // 7.4 Linear Regression Forecast
    println!("7.4 Linear Regression Forecast");

    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let t_array = Array::from_vec(t);

    let mean_t = t_array.mean();
    let mean_y = series.mean();

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let ti = t_array.get(&[i])?;
        let yi = series.get(&[i])?;
        numerator += (ti - mean_t) * (yi - mean_y);
        denominator += (ti - mean_t).powi(2);
    }

    let slope = numerator / denominator;
    let intercept = mean_y - slope * mean_t;

    println!("  Fitted model: y = {:.4}t + {:.4}", slope, intercept);
    println!("  Forecasts:");

    for h in 1..=horizon {
        let t_future = (n - 1 + h) as f64;
        let forecast = slope * t_future + intercept;
        println!("    h={} (t={}): {:.4}", h, n - 1 + h, forecast);
    }
    println!();

    // 7.5 Forecast Evaluation (in-sample)
    println!("7.5 Forecast Evaluation (In-Sample)");

    let mut mse_naive = 0.0;
    let mut mae_naive = 0.0;

    for i in 1..n {
        let actual = series.get(&[i])?;
        let forecast = series.get(&[i - 1])?; // Naive forecast
        let error = actual - forecast;

        mse_naive += error * error;
        mae_naive += error.abs();
    }

    mse_naive /= (n - 1) as f64;
    mae_naive /= (n - 1) as f64;

    println!("  Naive Method:");
    println!("    MSE: {:.4}", mse_naive);
    println!("    RMSE: {:.4}", mse_naive.sqrt());
    println!("    MAE: {:.4}", mae_naive);
    println!();

    println!("✓ Example 7 completed\n");
    Ok(())
}
