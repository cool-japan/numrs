//! # Preprocessing Pipeline
//!
//! Input validation, normalization, and feature extraction pipeline for ML serving.

use super::{Result, ServingError};
use crate::array::Array;

/// Preprocessing stage trait
pub trait PreprocessingStage: Send + Sync {
    /// Apply preprocessing stage
    fn apply(&self, input: &Array<f64>) -> Result<Array<f64>>;

    /// Get stage name
    fn name(&self) -> &str;

    /// Validate configuration
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Input validator for shape and value constraints
pub struct InputValidator {
    name: String,
    expected_shape: Vec<Option<usize>>,
    min_value: Option<f64>,
    max_value: Option<f64>,
    allow_nan: bool,
    allow_inf: bool,
}

impl InputValidator {
    /// Create new input validator
    pub fn new(expected_shape: Vec<Option<usize>>) -> Self {
        Self {
            name: "input_validator".to_string(),
            expected_shape,
            min_value: None,
            max_value: None,
            allow_nan: false,
            allow_inf: false,
        }
    }

    /// Set value range constraints
    pub fn with_value_range(mut self, min: f64, max: f64) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Set NaN handling
    pub fn with_nan_handling(mut self, allow: bool) -> Self {
        self.allow_nan = allow;
        self
    }

    /// Set infinity handling
    pub fn with_inf_handling(mut self, allow: bool) -> Self {
        self.allow_inf = allow;
        self
    }
}

impl PreprocessingStage for InputValidator {
    fn apply(&self, input: &Array<f64>) -> Result<Array<f64>> {
        // Validate shape
        let shape = input.shape();
        if self.expected_shape.len() != shape.len() {
            return Err(ServingError::InvalidShape {
                expected: self.expected_shape.clone(),
                actual: shape.clone(),
            });
        }

        for (expected, actual) in self.expected_shape.iter().zip(shape.iter()) {
            if let Some(exp_size) = expected {
                if exp_size != actual {
                    return Err(ServingError::InvalidShape {
                        expected: self.expected_shape.clone(),
                        actual: shape.clone(),
                    });
                }
            }
        }

        // Validate values
        let data = input.to_vec();
        for (i, &value) in data.iter().enumerate() {
            // Check NaN
            if value.is_nan() && !self.allow_nan {
                return Err(ServingError::ValidationError {
                    field: format!("element[{}]", i),
                    message: "NaN values not allowed".to_string(),
                });
            }

            // Check infinity
            if value.is_infinite() && !self.allow_inf {
                return Err(ServingError::ValidationError {
                    field: format!("element[{}]", i),
                    message: "Infinite values not allowed".to_string(),
                });
            }

            // Check range
            if let Some(min) = self.min_value {
                if value < min {
                    return Err(ServingError::ValidationError {
                        field: format!("element[{}]", i),
                        message: format!("Value {} below minimum {}", value, min),
                    });
                }
            }

            if let Some(max) = self.max_value {
                if value > max {
                    return Err(ServingError::ValidationError {
                        field: format!("element[{}]", i),
                        message: format!("Value {} above maximum {}", value, max),
                    });
                }
            }
        }

        Ok(input.clone())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Normalization methods
#[derive(Debug, Clone)]
pub enum NormalizationType {
    /// Min-max normalization to [0, 1]
    MinMax { min: f64, max: f64 },

    /// Z-score normalization (standardization)
    ZScore { mean: f64, std: f64 },

    /// L1 normalization
    L1,

    /// L2 normalization
    L2,
}

/// Normalizer for input preprocessing
pub struct Normalizer {
    name: String,
    normalization_type: NormalizationType,
}

impl Normalizer {
    /// Create min-max normalizer
    pub fn min_max(min: f64, max: f64) -> Result<Self> {
        if min >= max {
            return Err(ServingError::ValidationError {
                field: "min_max_range".to_string(),
                message: "min must be less than max".to_string(),
            });
        }

        Ok(Self {
            name: "normalizer_minmax".to_string(),
            normalization_type: NormalizationType::MinMax { min, max },
        })
    }

    /// Create z-score normalizer
    pub fn z_score(mean: f64, std: f64) -> Result<Self> {
        if std <= 0.0 {
            return Err(ServingError::ValidationError {
                field: "std".to_string(),
                message: "Standard deviation must be positive".to_string(),
            });
        }

        Ok(Self {
            name: "normalizer_zscore".to_string(),
            normalization_type: NormalizationType::ZScore { mean, std },
        })
    }

    /// Create L1 normalizer
    pub fn l1() -> Self {
        Self {
            name: "normalizer_l1".to_string(),
            normalization_type: NormalizationType::L1,
        }
    }

    /// Create L2 normalizer
    pub fn l2() -> Self {
        Self {
            name: "normalizer_l2".to_string(),
            normalization_type: NormalizationType::L2,
        }
    }
}

impl PreprocessingStage for Normalizer {
    fn apply(&self, input: &Array<f64>) -> Result<Array<f64>> {
        match &self.normalization_type {
            NormalizationType::MinMax { min, max } => {
                let data = input.to_vec();
                let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
                let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                if (data_max - data_min).abs() < 1e-10 {
                    let shape = input.shape().to_vec();
                    return Ok(Array::zeros(&shape));
                }

                let normalized: Vec<f64> = data
                    .iter()
                    .map(|&x| {
                        let scaled = (x - data_min) / (data_max - data_min);
                        scaled * (max - min) + min
                    })
                    .collect();

                let shape = input.shape().to_vec();
                Ok(Array::from_vec_shape(normalized, &shape)?)
            }

            NormalizationType::ZScore { mean, std } => {
                let normalized = input.subtract_scalar(*mean).divide_scalar(*std);
                Ok(normalized)
            }

            NormalizationType::L1 => {
                let data = input.to_vec();
                let l1_norm: f64 = data.iter().map(|x| x.abs()).sum();

                if l1_norm < 1e-10 {
                    return Ok(input.clone());
                }

                let normalized = input.divide_scalar(l1_norm);
                Ok(normalized)
            }

            NormalizationType::L2 => {
                let data = input.to_vec();
                let l2_norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();

                if l2_norm < 1e-10 {
                    return Ok(input.clone());
                }

                let normalized = input.divide_scalar(l2_norm);
                Ok(normalized)
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Feature extraction stage
pub struct FeatureExtractor {
    name: String,
    feature_indices: Vec<usize>,
}

impl FeatureExtractor {
    /// Create new feature extractor
    pub fn new(feature_indices: Vec<usize>) -> Self {
        Self {
            name: "feature_extractor".to_string(),
            feature_indices,
        }
    }
}

impl PreprocessingStage for FeatureExtractor {
    fn apply(&self, input: &Array<f64>) -> Result<Array<f64>> {
        let data = input.to_vec();
        let shape = input.shape();

        // For 1D input
        if shape.len() == 1 {
            let extracted: Vec<f64> = self
                .feature_indices
                .iter()
                .filter_map(|&idx| {
                    if idx < data.len() {
                        Some(data[idx])
                    } else {
                        None
                    }
                })
                .collect();

            if extracted.len() != self.feature_indices.len() {
                return Err(ServingError::ValidationError {
                    field: "feature_indices".to_string(),
                    message: "Some feature indices out of bounds".to_string(),
                });
            }

            return Ok(Array::from_vec(extracted));
        }

        // For 2D input (batch, features)
        if shape.len() == 2 {
            let batch_size = shape[0];
            let n_features = shape[1];

            let mut extracted = Vec::new();
            for i in 0..batch_size {
                for &idx in &self.feature_indices {
                    if idx >= n_features {
                        return Err(ServingError::ValidationError {
                            field: "feature_indices".to_string(),
                            message: format!("Feature index {} out of bounds", idx),
                        });
                    }
                    extracted.push(data[i * n_features + idx]);
                }
            }

            let new_shape = vec![batch_size, self.feature_indices.len()];
            return Ok(Array::from_vec_shape(extracted, &new_shape)?);
        }

        Err(ServingError::PreprocessingError {
            stage: "feature_extraction".to_string(),
            message: "Only 1D and 2D arrays supported".to_string(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Preprocessing pipeline
pub struct PreprocessingPipeline {
    stages: Vec<Box<dyn PreprocessingStage>>,
    /// Caching was never implemented: no setter exists to turn this on, and
    /// `apply()` below runs every stage unconditionally with no cache store
    /// or lookup. Kept rather than deleted since it documents the shape a
    /// real caching feature would need to fill in.
    #[allow(dead_code)]
    cache_enabled: bool,
}

impl PreprocessingPipeline {
    /// Create new preprocessing pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            cache_enabled: false,
        }
    }

    /// Add preprocessing stage
    pub fn add_stage(&mut self, stage: Box<dyn PreprocessingStage>) -> Result<()> {
        stage.validate()?;
        self.stages.push(stage);
        Ok(())
    }

    /// Add validator stage
    pub fn add_validator(&mut self, expected_shape: Vec<Option<usize>>) -> Result<()> {
        let validator = Box::new(InputValidator::new(expected_shape));
        self.add_stage(validator)
    }

    /// Add normalizer stage
    pub fn add_normalizer(&mut self, norm_type: NormalizationType) -> Result<()> {
        let normalizer = match norm_type {
            NormalizationType::MinMax { min, max } => Box::new(Normalizer::min_max(min, max)?),
            NormalizationType::ZScore { mean, std } => Box::new(Normalizer::z_score(mean, std)?),
            NormalizationType::L1 => Box::new(Normalizer::l1()),
            NormalizationType::L2 => Box::new(Normalizer::l2()),
        };
        self.add_stage(normalizer)
    }

    /// Add feature extractor stage
    pub fn add_feature_extractor(&mut self, feature_indices: Vec<usize>) -> Result<()> {
        let extractor = Box::new(FeatureExtractor::new(feature_indices));
        self.add_stage(extractor)
    }

    /// Apply all preprocessing stages
    pub fn apply(&self, input: &Array<f64>) -> Result<Array<f64>> {
        let mut current = input.clone();

        for (i, stage) in self.stages.iter().enumerate() {
            current = stage.apply(&current).map_err(|e| match e {
                ServingError::PreprocessingError { .. } => e,
                _ => ServingError::PreprocessingError {
                    stage: format!("stage_{}: {}", i, stage.name()),
                    message: format!("{}", e),
                },
            })?;
        }

        Ok(current)
    }

    /// Get number of stages
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Clear all stages
    pub fn clear(&mut self) {
        self.stages.clear();
    }
}

impl Default for PreprocessingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_validator_valid_shape() {
        let validator = InputValidator::new(vec![Some(2), Some(3)]);
        let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);

        let result = validator.apply(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_input_validator_invalid_shape() {
        let validator = InputValidator::new(vec![Some(2), Some(3)]);
        let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);

        let result = validator.apply(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_input_validator_value_range() {
        let validator = InputValidator::new(vec![None]).with_value_range(0.0, 10.0);

        let valid_input = Array::from_vec(vec![1.0, 5.0, 9.0]);
        assert!(validator.apply(&valid_input).is_ok());

        let invalid_input = Array::from_vec(vec![1.0, 15.0, 9.0]);
        assert!(validator.apply(&invalid_input).is_err());
    }

    #[test]
    fn test_input_validator_nan_handling() {
        let validator_disallow = InputValidator::new(vec![None]).with_nan_handling(false);

        let input_with_nan = Array::from_vec(vec![1.0, f64::NAN, 3.0]);
        assert!(validator_disallow.apply(&input_with_nan).is_err());

        let validator_allow = InputValidator::new(vec![None]).with_nan_handling(true);
        assert!(validator_allow.apply(&input_with_nan).is_ok());
    }

    #[test]
    fn test_normalizer_minmax() {
        let normalizer = Normalizer::min_max(0.0, 1.0).expect("Normalizer creation should succeed");

        let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let output = normalizer
            .apply(&input)
            .expect("Normalization should succeed");

        let data = output.to_vec();
        assert!((data[0] - 0.0).abs() < 1e-10);
        assert!((data[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_zscore() {
        let normalizer = Normalizer::z_score(0.0, 1.0).expect("Normalizer creation should succeed");

        let input = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let output = normalizer
            .apply(&input)
            .expect("Normalization should succeed");

        assert_eq!(output.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_normalizer_l1() {
        let normalizer = Normalizer::l1();

        let input = Array::from_vec(vec![3.0, 4.0]);
        let output = normalizer
            .apply(&input)
            .expect("Normalization should succeed");

        let l1_norm: f64 = output.to_vec().iter().map(|x| x.abs()).sum();
        assert!((l1_norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_l2() {
        let normalizer = Normalizer::l2();

        let input = Array::from_vec(vec![3.0, 4.0]);
        let output = normalizer
            .apply(&input)
            .expect("Normalization should succeed");

        let l2_norm: f64 = output.to_vec().iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((l2_norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_extractor_1d() {
        let extractor = FeatureExtractor::new(vec![0, 2, 4]);

        let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let output = extractor
            .apply(&input)
            .expect("Feature extraction should succeed");

        assert_eq!(output.to_vec(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_feature_extractor_2d() {
        let extractor = FeatureExtractor::new(vec![0, 2]);

        let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        let output = extractor
            .apply(&input)
            .expect("Feature extraction should succeed");

        assert_eq!(output.shape(), vec![2, 2]);
        assert_eq!(output.to_vec(), vec![1.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = PreprocessingPipeline::new();
        assert_eq!(pipeline.stage_count(), 0);
    }

    #[test]
    fn test_pipeline_add_stages() {
        let mut pipeline = PreprocessingPipeline::new();

        pipeline
            .add_validator(vec![None, Some(3)])
            .expect("Add validator should succeed");

        pipeline
            .add_normalizer(NormalizationType::L2)
            .expect("Add normalizer should succeed");

        assert_eq!(pipeline.stage_count(), 2);
    }

    #[test]
    fn test_pipeline_apply() {
        let mut pipeline = PreprocessingPipeline::new();

        pipeline
            .add_normalizer(NormalizationType::MinMax { min: 0.0, max: 1.0 })
            .expect("Add normalizer should succeed");

        let input = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let output = pipeline
            .apply(&input)
            .expect("Pipeline apply should succeed");

        let data = output.to_vec();
        assert!((data[0] - 0.0).abs() < 1e-10);
        assert!((data[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_clear() {
        let mut pipeline = PreprocessingPipeline::new();

        pipeline
            .add_validator(vec![None])
            .expect("Add validator should succeed");

        assert_eq!(pipeline.stage_count(), 1);

        pipeline.clear();
        assert_eq!(pipeline.stage_count(), 0);
    }
}
