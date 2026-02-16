//! # Production ML Serving Module
//!
//! This module provides production-ready machine learning serving capabilities for NumRS2,
//! including optimized inference, model management, preprocessing pipelines, and monitoring.
//!
//! ## Overview
//!
//! The serving module offers comprehensive infrastructure for deploying and serving ML models:
//!
//! - **Inference Engine**: Optimized forward pass execution with batch processing and caching
//! - **Model Registry**: Multi-model management with versioning and hot reloading
//! - **Preprocessing Pipeline**: Input validation, normalization, and feature extraction
//! - **Prediction API**: Synchronous, asynchronous, streaming, and batch predictions
//! - **Performance Optimization**: Model quantization, operator fusion, memory pooling
//! - **Monitoring**: Request latency, throughput, accuracy, and resource utilization tracking
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Prediction API                            │
//! │  (Sync/Async/Streaming/Batch)                                │
//! └────────────────────┬────────────────────────────────────────┘
//!                      │
//! ┌────────────────────┴────────────────────────────────────────┐
//! │              Preprocessing Pipeline                          │
//! │  (Validation → Normalization → Feature Extraction)           │
//! └────────────────────┬────────────────────────────────────────┘
//!                      │
//! ┌────────────────────┴────────────────────────────────────────┐
//! │                 Inference Engine                             │
//! │  (Model Warmup → Batch Processing → Caching)                 │
//! └────────────────────┬────────────────────────────────────────┘
//!                      │
//! ┌────────────────────┴────────────────────────────────────────┐
//! │                 Model Registry                               │
//! │  (Multi-model → Versioning → Hot Reloading)                  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## SCIRS2 Policy Compliance
//!
//! This module strictly follows SCIRS2 ecosystem policies:
//!
//! - **Array Operations**: ALWAYS use `scirs2_core::ndarray` (NEVER direct ndarray)
//! - **Random Numbers**: ALWAYS use `scirs2_core::random` (NEVER direct rand)
//! - **Parallel Processing**: ALWAYS use `scirs2_core::parallel_ops` (NEVER direct rayon)
//! - **Linear Algebra**: Use `scirs2_linalg` for matrix operations (Pure Rust via OxiBLAS)
//! - **Error Handling**: Base errors on `scirs2_core::error::CoreError`
//! - **Pure Rust**: 100% Pure Rust implementation (no C/C++ dependencies)
//!
//! ## Usage Examples
//!
//! ### Example 1: Basic Model Serving
//!
//! ```rust,ignore
//! use numrs2::new_modules::serving::{ModelRegistry, InferenceEngine, predict_sync};
//! use numrs2::prelude::*;
//!
//! // Create model registry and load model
//! let mut registry = ModelRegistry::new();
//! let model = /* load your model */;
//! registry.register("my_model", "v1.0", model)?;
//!
//! // Create inference engine
//! let engine = InferenceEngine::new(registry)?;
//!
//! // Make prediction
//! let input = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
//! let output = predict_sync(&engine, "my_model", &input)?;
//! ```
//!
//! ### Example 2: Batch Inference with Preprocessing
//!
//! ```rust,ignore
//! use numrs2::new_modules::serving::{PreprocessingPipeline, batch_predict};
//!
//! // Create preprocessing pipeline
//! let mut pipeline = PreprocessingPipeline::new();
//! pipeline.add_normalizer("min_max", 0.0, 1.0)?;
//! pipeline.add_validator("shape", vec![None, 3])?;
//!
//! // Batch prediction with preprocessing
//! let inputs = vec![input1, input2, input3];
//! let outputs = batch_predict(&engine, "my_model", &inputs, Some(&pipeline))?;
//! ```
//!
//! ### Example 3: Model Monitoring
//!
//! ```rust,ignore
//! use numrs2::new_modules::serving::{ServingMetrics, LatencyTracker};
//!
//! // Track metrics
//! let mut metrics = ServingMetrics::new();
//! let tracker = LatencyTracker::start();
//!
//! // Make prediction
//! let output = predict_sync(&engine, "my_model", &input)?;
//!
//! // Record metrics
//! metrics.record_latency(tracker.elapsed());
//! metrics.record_throughput(1);
//! ```
//!
//! ## Performance Considerations
//!
//! - **Batch Processing**: Dynamic batching for throughput optimization
//! - **Model Warmup**: Pre-execution for consistent latency
//! - **Memory Pooling**: Reusable buffers to reduce allocations
//! - **SIMD Optimization**: Vectorized operations where applicable
//! - **Operator Fusion**: Merged operations to reduce overhead
//! - **Quantization**: INT8/INT16 quantization for faster inference
//!
//! ## Thread Safety
//!
//! All components are designed for concurrent access:
//! - Model registry uses read-write locks for hot reloading
//! - Inference engine supports parallel batch processing
//! - Metrics collectors are thread-safe using atomic operations
//!
//! ## References
//!
//! - Crankshaw, D., et al. (2017). Clipper: A Low-Latency Online Prediction Serving System. *NSDI*.
//! - Olston, C., et al. (2017). TensorFlow-Serving: Flexible, High-Performance ML Serving. *KDD*.
//! - Moritz, P., et al. (2018). Ray: A Distributed Framework for Emerging AI Applications. *OSDI*.

use crate::array::Array;
use crate::error::NumRs2Error;
use std::fmt;

// Module declarations
pub mod inference;
pub mod metrics;
pub mod optimization;
pub mod predict;
pub mod preprocessing;
pub mod registry;

// Re-exports from submodules
pub use inference::*;
pub use metrics::*;
pub use optimization::*;
pub use predict::*;
pub use preprocessing::*;
pub use registry::*;

/// Result type for serving operations
pub type Result<T> = std::result::Result<T, ServingError>;

/// Comprehensive error type for ML serving operations
#[derive(Debug, Clone)]
pub enum ServingError {
    /// Model not found in registry
    ModelNotFound {
        model_name: String,
        version: Option<String>,
    },

    /// Invalid model version
    InvalidVersion {
        model_name: String,
        version: String,
        message: String,
    },

    /// Model loading error
    ModelLoadError { model_name: String, message: String },

    /// Inference error during prediction
    InferenceError { model_name: String, message: String },

    /// Input validation error
    ValidationError { field: String, message: String },

    /// Preprocessing error
    PreprocessingError { stage: String, message: String },

    /// Invalid input shape
    InvalidShape {
        expected: Vec<Option<usize>>,
        actual: Vec<usize>,
    },

    /// Batch size mismatch
    BatchSizeMismatch { expected: usize, actual: usize },

    /// Quantization error
    QuantizationError { message: String },

    /// Memory pool exhausted
    MemoryPoolExhausted { requested: usize, available: usize },

    /// Timeout error
    TimeoutError { operation: String, timeout_ms: u64 },

    /// Concurrency error
    ConcurrencyError { message: String },

    /// Metrics collection error
    MetricsError { message: String },

    /// Integration error with NumRS2
    NumRs2IntegrationError { source: Box<NumRs2Error> },

    /// Generic error with custom message
    Other { message: String },
}

impl fmt::Display for ServingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServingError::ModelNotFound {
                model_name,
                version,
            } => {
                if let Some(v) = version {
                    write!(f, "Model '{}' version '{}' not found", model_name, v)
                } else {
                    write!(f, "Model '{}' not found", model_name)
                }
            }
            ServingError::InvalidVersion {
                model_name,
                version,
                message,
            } => {
                write!(
                    f,
                    "Invalid version '{}' for model '{}': {}",
                    version, model_name, message
                )
            }
            ServingError::ModelLoadError {
                model_name,
                message,
            } => {
                write!(f, "Failed to load model '{}': {}", model_name, message)
            }
            ServingError::InferenceError {
                model_name,
                message,
            } => {
                write!(f, "Inference error in model '{}': {}", model_name, message)
            }
            ServingError::ValidationError { field, message } => {
                write!(f, "Validation error for field '{}': {}", field, message)
            }
            ServingError::PreprocessingError { stage, message } => {
                write!(f, "Preprocessing error in stage '{}': {}", stage, message)
            }
            ServingError::InvalidShape { expected, actual } => {
                write!(
                    f,
                    "Invalid shape: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            ServingError::BatchSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Batch size mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            ServingError::QuantizationError { message } => {
                write!(f, "Quantization error: {}", message)
            }
            ServingError::MemoryPoolExhausted {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Memory pool exhausted: requested {} bytes, available {} bytes",
                    requested, available
                )
            }
            ServingError::TimeoutError {
                operation,
                timeout_ms,
            } => {
                write!(
                    f,
                    "Operation '{}' timed out after {} ms",
                    operation, timeout_ms
                )
            }
            ServingError::ConcurrencyError { message } => {
                write!(f, "Concurrency error: {}", message)
            }
            ServingError::MetricsError { message } => {
                write!(f, "Metrics error: {}", message)
            }
            ServingError::NumRs2IntegrationError { source } => {
                write!(f, "NumRS2 integration error: {}", source)
            }
            ServingError::Other { message } => {
                write!(f, "Serving error: {}", message)
            }
        }
    }
}

impl std::error::Error for ServingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ServingError::NumRs2IntegrationError { source } => Some(source),
            _ => None,
        }
    }
}

impl From<NumRs2Error> for ServingError {
    fn from(error: NumRs2Error) -> Self {
        ServingError::NumRs2IntegrationError {
            source: Box::new(error),
        }
    }
}

/// Helper function to validate array shape matches expected pattern
///
/// # Arguments
///
/// * `expected` - Expected shape pattern (None means any size for that dimension)
/// * `actual` - Actual shape
///
/// # Returns
///
/// * `Ok(())` if shape matches
/// * `Err(ServingError)` if shape doesn't match
pub fn validate_shape(expected: &[Option<usize>], actual: &[usize]) -> Result<()> {
    if expected.len() != actual.len() {
        return Err(ServingError::InvalidShape {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        });
    }

    for (i, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
        if let Some(exp_size) = exp {
            if exp_size != act {
                return Err(ServingError::InvalidShape {
                    expected: expected.to_vec(),
                    actual: actual.to_vec(),
                });
            }
        }
    }

    Ok(())
}

/// Helper function to validate batch size
pub fn validate_batch_size(expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(ServingError::BatchSizeMismatch { expected, actual });
    }
    Ok(())
}

#[cfg(test)]
mod module_tests {
    use super::*;

    #[test]
    fn test_validate_shape_exact_match() {
        let expected = vec![Some(2), Some(3)];
        let actual = vec![2, 3];
        assert!(validate_shape(&expected, &actual).is_ok());
    }

    #[test]
    fn test_validate_shape_with_none() {
        let expected = vec![None, Some(3)];
        let actual = vec![5, 3];
        assert!(validate_shape(&expected, &actual).is_ok());
    }

    #[test]
    fn test_validate_shape_mismatch() {
        let expected = vec![Some(2), Some(3)];
        let actual = vec![2, 4];
        assert!(validate_shape(&expected, &actual).is_err());
    }

    #[test]
    fn test_validate_shape_dimension_mismatch() {
        let expected = vec![Some(2), Some(3)];
        let actual = vec![2];
        assert!(validate_shape(&expected, &actual).is_err());
    }

    #[test]
    fn test_validate_batch_size_match() {
        assert!(validate_batch_size(32, 32).is_ok());
    }

    #[test]
    fn test_validate_batch_size_mismatch() {
        assert!(validate_batch_size(32, 16).is_err());
    }

    #[test]
    fn test_error_display() {
        let err = ServingError::ModelNotFound {
            model_name: "test_model".to_string(),
            version: Some("v1.0".to_string()),
        };
        let display = format!("{}", err);
        assert!(display.contains("test_model"));
        assert!(display.contains("v1.0"));
    }

    #[test]
    fn test_error_from_numrs2() {
        let numrs2_err = NumRs2Error::DimensionMismatch("test".to_string());
        let serving_err: ServingError = numrs2_err.into();

        match serving_err {
            ServingError::NumRs2IntegrationError { .. } => {}
            _ => panic!("Expected NumRs2IntegrationError"),
        }
    }
}
