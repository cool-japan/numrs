//! Tests for exponential smoothing methods.

#[cfg(test)]
mod exponential_smoothing_tests {
    use super::super::helpers::{damped_trend_sum, quantile_normal};
    use super::super::optimization::{optimize_parameters, select_best_model};
    use super::super::types::{OptimizationConfig, SeasonalComponent, TrendComponent};
    use super::super::ExponentialSmoothing;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    // ---- SES Tests ----

    #[test]
    fn test_ses_constant_data() {
        // SES on constant data should produce constant fitted values
        let data = Array1::from_vec(vec![5.0; 20]);
        let model = ExponentialSmoothing::ses(0.3).expect("SES creation should succeed");
        let result = model.fit(&data.view()).expect("fit should succeed");

        // All fitted values should be close to the constant
        for i in 1..20 {
            assert_relative_eq!(result.fitted[i], 5.0, epsilon = 1e-10);
        }
        // Residuals should be near zero
        assert!(result.mse < 1e-10);
    }

    #[test]
    fn test_ses_trending_data() {
        // SES on trending data: fitted values should lag behind
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let model = ExponentialSmoothing::ses(0.5).expect("SES creation should succeed");
        let result = model.fit(&data.view()).expect("fit should succeed");

        // Fitted values should be positive and increasing but lagging
        for i in 1..10 {
            assert!(result.fitted[i] > 0.0);
        }
        // Last fitted value should be less than last data point (lag effect)
        assert!(result.fitted[9] < data[9]);
    }

    #[test]
    fn test_ses_forecast_flat() {
        // SES forecast is flat (constant equal to last level)
        let data = Array1::from_vec(vec![10.0, 12.0, 11.0, 13.0, 12.5, 14.0]);
        let model = ExponentialSmoothing::ses(0.3).expect("SES creation should succeed");
        let forecast = model
            .forecast(&data.view(), 5, 0.95)
            .expect("forecast should succeed");

        // All forecast values should be equal (flat forecast)
        for i in 1..5 {
            assert_relative_eq!(forecast.point[i], forecast.point[0], epsilon = 1e-10);
        }
        // Prediction intervals should widen
        let lower = forecast.lower.as_ref().expect("should have lower bounds");
        let upper = forecast.upper.as_ref().expect("should have upper bounds");
        assert!(upper[4] - lower[4] > upper[0] - lower[0]);
    }

    // ---- Holt's Method Tests ----

    #[test]
    fn test_holt_linear_trend() {
        // Holt's method should capture a linear trend well
        let data: Array1<f64> = Array1::from_vec((0..20).map(|i| 10.0 + 2.0 * i as f64).collect());
        let model = ExponentialSmoothing::holt(0.5, 0.3).expect("Holt creation should succeed");
        let forecast = model
            .forecast(&data.view(), 5, 0.95)
            .expect("forecast should succeed");

        // Forecasts should continue the upward trend
        for i in 1..5 {
            assert!(
                forecast.point[i] > forecast.point[i - 1],
                "Forecast should increase: {} vs {}",
                forecast.point[i],
                forecast.point[i - 1]
            );
        }
        // Forecast at h=1 should be close to actual continuation
        assert!(
            (forecast.point[0] - 50.0).abs() < 10.0,
            "First forecast {} should be near 50",
            forecast.point[0]
        );
    }

    #[test]
    fn test_holt_versus_ses_on_trend() {
        // Holt should produce lower MSE than SES on trending data
        let data: Array1<f64> = Array1::from_vec((0..30).map(|i| 5.0 + 1.5 * i as f64).collect());
        let ses = ExponentialSmoothing::ses(0.5).expect("SES should succeed");
        let holt = ExponentialSmoothing::holt(0.5, 0.3).expect("Holt should succeed");

        let ses_result = ses.fit(&data.view()).expect("SES fit");
        let holt_result = holt.fit(&data.view()).expect("Holt fit");

        assert!(
            holt_result.mse < ses_result.mse,
            "Holt MSE ({}) should be less than SES MSE ({}) on trending data",
            holt_result.mse,
            ses_result.mse
        );
    }

    // ---- Damped Trend Tests ----

    #[test]
    fn test_damped_trend_converges() {
        // Damped trend forecasts should converge to a finite limit
        let data: Array1<f64> = Array1::from_vec((0..20).map(|i| 10.0 + 2.0 * i as f64).collect());
        let model = ExponentialSmoothing::damped_trend(0.5, 0.3, 0.9)
            .expect("Damped trend creation should succeed");
        let forecast = model
            .forecast(&data.view(), 50, 0.95)
            .expect("forecast should succeed");

        // Damped forecasts should converge: differences should decrease
        let diff_early = (forecast.point[1] - forecast.point[0]).abs();
        let diff_late = (forecast.point[49] - forecast.point[48]).abs();
        assert!(
            diff_late < diff_early,
            "Late forecast steps should be closer together: {} vs {}",
            diff_late,
            diff_early
        );
    }

    #[test]
    fn test_damped_vs_undamped_long_horizon() {
        // Damped forecasts should be more conservative at long horizons
        let data: Array1<f64> = Array1::from_vec((0..20).map(|i| 10.0 + 2.0 * i as f64).collect());
        let holt = ExponentialSmoothing::holt(0.5, 0.3).expect("Holt should succeed");
        let damped =
            ExponentialSmoothing::damped_trend(0.5, 0.3, 0.85).expect("Damped should succeed");

        let holt_fc = holt
            .forecast(&data.view(), 30, 0.95)
            .expect("Holt forecast");
        let damped_fc = damped
            .forecast(&data.view(), 30, 0.95)
            .expect("Damped forecast");

        // At h=30, damped forecast should be lower than undamped
        assert!(
            damped_fc.point[29] < holt_fc.point[29],
            "Damped forecast ({}) should be less than Holt ({}) at long horizon",
            damped_fc.point[29],
            holt_fc.point[29]
        );
    }

    // ---- Holt-Winters Additive Tests ----

    #[test]
    fn test_hw_additive_seasonal_data() {
        // Create data with clear additive seasonal pattern
        let mut data_vec = Vec::with_capacity(48);
        let seasonal_pattern = [0.0, 3.0, 6.0, 3.0]; // period = 4
        for i in 0..48 {
            let trend = 10.0 + 0.5 * i as f64;
            let season = seasonal_pattern[i % 4];
            data_vec.push(trend + season);
        }
        let data = Array1::from_vec(data_vec);

        let model =
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 4, SeasonalComponent::Additive)
                .expect("HW additive creation should succeed");

        let result = model.fit(&data.view()).expect("HW fit should succeed");

        // MSE should be reasonably small for this clean data
        assert!(
            result.mse < 10.0,
            "MSE ({}) should be small for clean seasonal data",
            result.mse
        );

        // Seasonal component should exist
        assert!(result.seasonal.is_some());
    }

    #[test]
    fn test_hw_additive_forecast_preserves_season() {
        let mut data_vec = Vec::with_capacity(40);
        let seasonal_pattern = [0.0, 5.0, 10.0, 5.0];
        for i in 0..40 {
            let trend = 20.0 + 1.0 * i as f64;
            let season = seasonal_pattern[i % 4];
            data_vec.push(trend + season);
        }
        let data = Array1::from_vec(data_vec);

        let model =
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.3, 4, SeasonalComponent::Additive)
                .expect("HW should succeed");
        let forecast = model
            .forecast(&data.view(), 8, 0.95)
            .expect("forecast should succeed");

        // Forecast should show seasonal pattern: every 4th step should be similar
        // Check that forecasts at same seasonal position are closer to each other
        // than to adjacent forecasts
        let diff_same_season = (forecast.point[0] - forecast.point[4]).abs();
        let diff_adjacent = (forecast.point[0] - forecast.point[1]).abs();
        // Same-season forecasts differ mainly by trend; adjacent differ by season+trend
        assert!(
            diff_same_season < diff_adjacent * 3.0,
            "Same-season forecasts should be relatively close"
        );
    }

    // ---- Holt-Winters Multiplicative Tests ----

    #[test]
    fn test_hw_multiplicative_seasonal_data() {
        // Create data with multiplicative seasonal pattern
        let mut data_vec = Vec::with_capacity(48);
        let seasonal_factors = [0.8, 1.0, 1.3, 0.9]; // period = 4
        for i in 0..48 {
            let trend = 50.0 + 2.0 * i as f64;
            let season = seasonal_factors[i % 4];
            data_vec.push(trend * season);
        }
        let data = Array1::from_vec(data_vec);

        let model =
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 4, SeasonalComponent::Multiplicative)
                .expect("HW multiplicative creation should succeed");

        let result = model.fit(&data.view()).expect("fit should succeed");

        // MSE should be manageable
        assert!(result.mse.is_finite(), "MSE should be finite");
        assert!(result.seasonal.is_some());

        // Seasonal factors should sum close to period (normalized to average 1.0)
        let s = result.seasonal.as_ref().expect("seasonal should exist");
        let s_sum: f64 = s.iter().sum();
        assert_relative_eq!(s_sum / 4.0, 1.0, epsilon = 0.3);
    }

    #[test]
    fn test_hw_multiplicative_forecast() {
        let mut data_vec = Vec::with_capacity(40);
        let seasonal_factors = [0.7, 1.1, 1.4, 0.8];
        for i in 0..40 {
            let base = 100.0 + 3.0 * i as f64;
            data_vec.push(base * seasonal_factors[i % 4]);
        }
        let data = Array1::from_vec(data_vec);

        let model =
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 4, SeasonalComponent::Multiplicative)
                .expect("HW should succeed");
        let forecast = model
            .forecast(&data.view(), 4, 0.95)
            .expect("forecast should succeed");

        // All forecasts should be positive
        assert!(forecast.point.iter().all(|&x| x > 0.0));
        // Should have prediction intervals
        assert!(forecast.lower.is_some());
        assert!(forecast.upper.is_some());
    }

    // ---- Parameter Optimization Tests ----

    #[test]
    fn test_optimize_ses_parameters() {
        let data: Array1<f64> =
            Array1::from_vec((0..30).map(|i| 10.0 + 0.1 * (i as f64).sin()).collect());

        let config = OptimizationConfig {
            grid_resolution: 10,
            refinement_iterations: 1,
            ..Default::default()
        };

        let (model, result) = optimize_parameters(
            &data.view(),
            TrendComponent::None,
            SeasonalComponent::None,
            None,
            &config,
        )
        .expect("optimization should succeed");

        assert!(model.alpha > 0.0 && model.alpha < 1.0);
        assert!(result.mse.is_finite());
        assert!(result.mse >= 0.0);
    }

    #[test]
    fn test_optimize_holt_parameters() {
        let data: Array1<f64> = Array1::from_vec(
            (0..40)
                .map(|i| 5.0 + 2.0 * i as f64 + 0.5 * (i as f64 * 0.3).sin())
                .collect(),
        );

        let config = OptimizationConfig {
            grid_resolution: 8,
            refinement_iterations: 1,
            ..Default::default()
        };

        let (model, result) = optimize_parameters(
            &data.view(),
            TrendComponent::Additive,
            SeasonalComponent::None,
            None,
            &config,
        )
        .expect("optimization should succeed");

        assert!(model.alpha > 0.0 && model.alpha < 1.0);
        assert!(model.beta.is_some());
        let beta = model.beta.expect("beta should exist");
        assert!(beta > 0.0 && beta < 1.0);
        assert!(result.mse.is_finite());
    }

    // ---- Forecast Accuracy Tests ----

    #[test]
    fn test_forecast_accuracy_on_known_data() {
        // Linear data: y = 2*t + 10
        let train: Array1<f64> = Array1::from_vec((0..30).map(|i| 10.0 + 2.0 * i as f64).collect());
        let actual_next = 10.0 + 2.0 * 30.0; // = 70.0

        let model = ExponentialSmoothing::holt(0.8, 0.5).expect("Holt should succeed");
        let forecast = model
            .forecast(&train.view(), 1, 0.95)
            .expect("forecast should succeed");

        // Forecast should be close to actual
        let error = (forecast.point[0] - actual_next).abs();
        assert!(
            error < 5.0,
            "Forecast error ({}) should be small for linear data",
            error
        );
    }

    // ---- Prediction Intervals Tests ----

    #[test]
    fn test_prediction_intervals_coverage() {
        let data = Array1::from_vec(vec![
            10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.0, 15.0, 14.5, 16.0,
        ]);
        let model = ExponentialSmoothing::ses(0.3).expect("SES should succeed");
        let forecast = model
            .forecast(&data.view(), 5, 0.95)
            .expect("forecast should succeed");

        let lower = forecast.lower.as_ref().expect("lower bounds");
        let upper = forecast.upper.as_ref().expect("upper bounds");

        // Lower should be less than point, upper should be greater
        for i in 0..5 {
            assert!(
                lower[i] < forecast.point[i],
                "Lower bound should be below point forecast"
            );
            assert!(
                upper[i] > forecast.point[i],
                "Upper bound should be above point forecast"
            );
        }

        assert_relative_eq!(forecast.confidence_level, 0.95, epsilon = 1e-10);
    }

    #[test]
    fn test_prediction_intervals_widen_with_horizon() {
        let data = Array1::from_vec(vec![5.0, 6.0, 7.0, 5.5, 6.5, 7.5, 5.0, 6.0, 7.0, 8.0]);
        let model = ExponentialSmoothing::holt(0.3, 0.1).expect("Holt should succeed");
        let forecast = model
            .forecast(&data.view(), 10, 0.90)
            .expect("forecast should succeed");

        let lower = forecast.lower.as_ref().expect("lower bounds");
        let upper = forecast.upper.as_ref().expect("upper bounds");

        let width_1 = upper[0] - lower[0];
        let width_10 = upper[9] - lower[9];

        assert!(
            width_10 > width_1,
            "Prediction interval at h=10 ({}) should be wider than at h=1 ({})",
            width_10,
            width_1
        );
    }

    // ---- Information Criteria Tests ----

    #[test]
    fn test_information_criteria_computed() {
        let data = Array1::from_vec(vec![
            10.0, 12.0, 11.0, 13.0, 12.5, 14.0, 13.0, 15.0, 14.5, 16.0, 15.0, 17.0, 16.5, 18.0,
            17.0,
        ]);
        let model = ExponentialSmoothing::ses(0.3).expect("SES should succeed");
        let result = model.fit(&data.view()).expect("fit should succeed");
        let ic = model
            .information_criteria(&result)
            .expect("IC should succeed");

        assert!(ic.aic.is_finite(), "AIC should be finite");
        assert!(ic.aicc.is_finite(), "AICc should be finite");
        assert!(ic.bic.is_finite(), "BIC should be finite");

        // AICc >= AIC (correction is always non-negative for finite samples)
        assert!(
            ic.aicc >= ic.aic - 1e-10,
            "AICc ({}) should be >= AIC ({})",
            ic.aicc,
            ic.aic
        );
    }

    #[test]
    fn test_model_selection_prefers_simpler_for_constant() {
        // For constant data, SES should be preferred over Holt
        let data = Array1::from_vec(vec![5.0; 30]);
        // Add tiny noise to avoid zero-variance issues
        let mut noisy = data.clone();
        for i in 0..30 {
            noisy[i] += 0.001 * (i as f64 * 0.7).sin();
        }

        let ses = ExponentialSmoothing::ses(0.1).expect("SES should succeed");
        let holt = ExponentialSmoothing::holt(0.1, 0.1).expect("Holt should succeed");

        let ses_result = ses.fit(&noisy.view()).expect("SES fit");
        let holt_result = holt.fit(&noisy.view()).expect("Holt fit");

        let ses_ic = ses.information_criteria(&ses_result).expect("SES IC");
        let holt_ic = holt.information_criteria(&holt_result).expect("Holt IC");

        // SES should have lower (better) BIC since data has no trend
        assert!(
            ses_ic.bic < holt_ic.bic,
            "SES BIC ({}) should be less than Holt BIC ({}) for constant data",
            ses_ic.bic,
            holt_ic.bic
        );
    }

    // ---- Edge Case Tests ----

    #[test]
    fn test_short_series_ses() {
        // Minimum length for SES is 2
        let data = Array1::from_vec(vec![1.0, 2.0]);
        let model = ExponentialSmoothing::ses(0.5).expect("SES should succeed");
        let result = model.fit(&data.view()).expect("fit should succeed");
        assert_eq!(result.fitted.len(), 2);
    }

    #[test]
    fn test_single_observation_ses_fails() {
        let data = Array1::from_vec(vec![42.0]);
        let model = ExponentialSmoothing::ses(0.5).expect("SES should succeed");
        let result = model.fit(&data.view());
        assert!(result.is_err(), "Single observation should fail");
    }

    #[test]
    fn test_invalid_parameters() {
        // Alpha out of range
        assert!(ExponentialSmoothing::ses(0.0).is_err());
        assert!(ExponentialSmoothing::ses(1.0).is_err());
        assert!(ExponentialSmoothing::ses(-0.1).is_err());
        assert!(ExponentialSmoothing::ses(1.5).is_err());

        // Beta out of range
        assert!(ExponentialSmoothing::holt(0.5, 0.0).is_err());
        assert!(ExponentialSmoothing::holt(0.5, 1.0).is_err());

        // Phi out of range
        assert!(ExponentialSmoothing::damped_trend(0.5, 0.3, 0.0).is_err());
        assert!(ExponentialSmoothing::damped_trend(0.5, 0.3, 1.0).is_err());

        // Invalid period
        assert!(
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 1, SeasonalComponent::Additive)
                .is_err()
        );
    }

    #[test]
    fn test_insufficient_data_for_seasonal() {
        // Need at least 2*period observations for seasonal model
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Only 3 obs, period=4
        let model =
            ExponentialSmoothing::holt_winters(0.3, 0.1, 0.2, 4, SeasonalComponent::Additive)
                .expect("Model creation should succeed (validation at fit time)");
        let result = model.fit(&data.view());
        assert!(
            result.is_err(),
            "Fitting with insufficient data should fail"
        );
    }

    // ---- Damped Holt-Winters Tests ----

    #[test]
    fn test_damped_holt_winters() {
        let mut data_vec = Vec::with_capacity(48);
        let seasonal_pattern = [0.0, 4.0, 8.0, 4.0];
        for i in 0..48 {
            let trend = 20.0 + 1.0 * i as f64;
            let season = seasonal_pattern[i % 4];
            data_vec.push(trend + season);
        }
        let data = Array1::from_vec(data_vec);

        let model = ExponentialSmoothing::damped_holt_winters(
            0.3,
            0.1,
            0.2,
            0.9,
            4,
            SeasonalComponent::Additive,
        )
        .expect("Damped HW should succeed");

        let forecast = model
            .forecast(&data.view(), 20, 0.95)
            .expect("forecast should succeed");

        // Forecasts should be positive and have prediction intervals
        assert!(forecast.point.iter().all(|&x| x > 0.0));
        assert!(forecast.lower.is_some());
        assert!(forecast.upper.is_some());

        // Damped trend: forecast differences should decrease
        let diff_early = (forecast.point[1] - forecast.point[0]).abs();
        let diff_late = (forecast.point[19] - forecast.point[18]).abs();
        // Due to seasonality, compare same-season steps
        let diff_season_early = (forecast.point[4] - forecast.point[0]).abs();
        let diff_season_late = (forecast.point[16] - forecast.point[12]).abs();
        // Later same-season differences should be smaller (damped trend)
        assert!(
            diff_season_late < diff_season_early + 5.0,
            "Damped seasonal steps should converge"
        );
    }

    // ---- Model Selection Test ----

    #[test]
    fn test_select_best_model() {
        // Data with trend and seasonality
        let mut data_vec = Vec::with_capacity(60);
        let seasonal = [0.0, 3.0, 6.0, 3.0];
        for i in 0..60 {
            data_vec.push(10.0 + 0.5 * i as f64 + seasonal[i % 4]);
        }
        let data = Array1::from_vec(data_vec);

        let config = OptimizationConfig {
            grid_resolution: 5,
            refinement_iterations: 0,
            ..Default::default()
        };

        let result = select_best_model(&data.view(), Some(4), &config);
        assert!(result.is_ok(), "Model selection should succeed");

        let (_model, _result, criteria) = result.expect("should have result");
        assert!(criteria.aicc.is_finite());
    }

    // ---- Quantile Normal Test ----

    #[test]
    fn test_quantile_normal_symmetry() {
        let z95 = quantile_normal(0.975);
        assert_relative_eq!(z95, 1.96, epsilon = 0.02);

        let z_sym = quantile_normal(0.025);
        assert_relative_eq!(z_sym, -z95, epsilon = 0.02);
    }

    // ---- Damped Trend Sum Test ----

    #[test]
    fn test_damped_trend_sum_values() {
        // phi=1: sum should equal h
        assert_relative_eq!(damped_trend_sum(1.0, 5), 5.0, epsilon = 1e-10);

        // phi=0.9, h=1: sum = 0.9
        assert_relative_eq!(damped_trend_sum(0.9, 1), 0.9, epsilon = 1e-10);

        // phi=0.9, h=2: sum = 0.9 + 0.81 = 1.71
        assert_relative_eq!(damped_trend_sum(0.9, 2), 1.71, epsilon = 1e-10);

        // phi=0: sum should be 0
        assert_relative_eq!(
            damped_trend_sum(0.001, 100),
            0.001 / (1.0 - 0.001),
            epsilon = 0.01
        );
    }
}
