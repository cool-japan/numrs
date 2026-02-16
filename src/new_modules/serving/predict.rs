//! # Prediction API
//!
//! Synchronous, asynchronous, streaming, and batch prediction endpoints.

use super::{InferenceEngine, PreprocessingPipeline, Result, ServingError};
use crate::array::Array;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Prediction request
#[derive(Clone)]
pub struct PredictionRequest {
    /// Input data
    pub input: Array<f64>,

    /// Request ID (optional)
    pub request_id: Option<String>,

    /// Preprocessing enabled
    pub preprocess: bool,

    /// Timeout in milliseconds (optional)
    pub timeout_ms: Option<u64>,
}

impl PredictionRequest {
    /// Create new prediction request
    pub fn new(input: Array<f64>) -> Self {
        Self {
            input,
            request_id: None,
            preprocess: true,
            timeout_ms: None,
        }
    }

    /// Set request ID
    pub fn with_id(mut self, id: String) -> Self {
        self.request_id = Some(id);
        self
    }

    /// Disable preprocessing
    pub fn without_preprocessing(mut self) -> Self {
        self.preprocess = false;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

/// Prediction response
#[derive(Clone)]
pub struct PredictionResponse {
    /// Output data
    pub output: Array<f64>,

    /// Request ID (if provided)
    pub request_id: Option<String>,

    /// Prediction latency in milliseconds
    pub latency_ms: f64,

    /// Preprocessing latency in milliseconds
    pub preprocessing_ms: f64,

    /// Inference latency in milliseconds
    pub inference_ms: f64,
}

impl PredictionResponse {
    /// Create new prediction response
    pub fn new(output: Array<f64>) -> Self {
        Self {
            output,
            request_id: None,
            latency_ms: 0.0,
            preprocessing_ms: 0.0,
            inference_ms: 0.0,
        }
    }
}

/// Batch prediction request
pub struct BatchPredictionRequest {
    /// Input batch
    pub inputs: Vec<Array<f64>>,

    /// Request IDs (optional)
    pub request_ids: Option<Vec<String>>,

    /// Preprocessing enabled
    pub preprocess: bool,

    /// Batch timeout in milliseconds (optional)
    pub timeout_ms: Option<u64>,
}

impl BatchPredictionRequest {
    /// Create new batch prediction request
    pub fn new(inputs: Vec<Array<f64>>) -> Self {
        Self {
            inputs,
            request_ids: None,
            preprocess: true,
            timeout_ms: None,
        }
    }

    /// Set request IDs
    pub fn with_ids(mut self, ids: Vec<String>) -> Self {
        self.request_ids = Some(ids);
        self
    }

    /// Disable preprocessing
    pub fn without_preprocessing(mut self) -> Self {
        self.preprocess = false;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

/// Batch prediction response
pub struct BatchPredictionResponse {
    /// Output batch
    pub outputs: Vec<Array<f64>>,

    /// Request IDs (if provided)
    pub request_ids: Option<Vec<String>>,

    /// Total latency in milliseconds
    pub total_latency_ms: f64,

    /// Average per-sample latency in milliseconds
    pub avg_latency_ms: f64,

    /// Throughput (samples per second)
    pub throughput: f64,
}

/// Synchronous prediction
pub fn predict_sync(
    engine: &InferenceEngine,
    input: &Array<f64>,
    pipeline: Option<&PreprocessingPipeline>,
) -> Result<PredictionResponse> {
    let start = Instant::now();

    // Preprocessing
    let preprocessing_start = Instant::now();
    let processed_input = if let Some(p) = pipeline {
        p.apply(input)?
    } else {
        input.clone()
    };
    let preprocessing_ms = preprocessing_start.elapsed().as_secs_f64() * 1000.0;

    // Inference
    let inference_start = Instant::now();
    let output = engine.infer(&processed_input)?;
    let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(PredictionResponse {
        output,
        request_id: None,
        latency_ms,
        preprocessing_ms,
        inference_ms,
    })
}

/// Batch prediction
pub fn predict_batch(
    engine: &InferenceEngine,
    inputs: &[Array<f64>],
    pipeline: Option<&PreprocessingPipeline>,
) -> Result<BatchPredictionResponse> {
    let start = Instant::now();

    if inputs.is_empty() {
        return Ok(BatchPredictionResponse {
            outputs: Vec::new(),
            request_ids: None,
            total_latency_ms: 0.0,
            avg_latency_ms: 0.0,
            throughput: 0.0,
        });
    }

    // Preprocessing
    let processed_inputs: Vec<Array<f64>> = if let Some(p) = pipeline {
        inputs
            .iter()
            .map(|input| p.apply(input))
            .collect::<Result<Vec<_>>>()?
    } else {
        inputs.to_vec()
    };

    // Inference
    let outputs = engine.infer_batch(&processed_inputs)?;

    let total_latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let avg_latency_ms = total_latency_ms / inputs.len() as f64;
    let throughput = (inputs.len() as f64 / total_latency_ms) * 1000.0;

    Ok(BatchPredictionResponse {
        outputs,
        request_ids: None,
        total_latency_ms,
        avg_latency_ms,
        throughput,
    })
}

/// Streaming prediction iterator
pub struct StreamingPredictor<'a> {
    engine: &'a InferenceEngine,
    pipeline: Option<&'a PreprocessingPipeline>,
    inputs: Vec<Array<f64>>,
    current_index: usize,
}

impl<'a> StreamingPredictor<'a> {
    /// Create new streaming predictor
    pub fn new(
        engine: &'a InferenceEngine,
        inputs: Vec<Array<f64>>,
        pipeline: Option<&'a PreprocessingPipeline>,
    ) -> Self {
        Self {
            engine,
            pipeline,
            inputs,
            current_index: 0,
        }
    }

    /// Get total number of inputs
    pub fn total(&self) -> usize {
        self.inputs.len()
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.current_index
    }

    /// Check if completed
    pub fn is_complete(&self) -> bool {
        self.current_index >= self.inputs.len()
    }
}

impl<'a> Iterator for StreamingPredictor<'a> {
    type Item = Result<PredictionResponse>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.inputs.len() {
            return None;
        }

        let input = &self.inputs[self.current_index];
        self.current_index += 1;

        Some(predict_sync(self.engine, input, self.pipeline))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.inputs.len().saturating_sub(self.current_index);
        (remaining, Some(remaining))
    }
}

/// Prediction with timeout
pub fn predict_with_timeout(
    engine: &InferenceEngine,
    input: &Array<f64>,
    pipeline: Option<&PreprocessingPipeline>,
    timeout_ms: u64,
) -> Result<PredictionResponse> {
    let start = Instant::now();

    // Simple timeout check (in a real implementation, this would be async)
    let result = predict_sync(engine, input, pipeline)?;

    let elapsed_ms = start.elapsed().as_millis() as u64;
    if elapsed_ms > timeout_ms {
        return Err(ServingError::TimeoutError {
            operation: "prediction".to_string(),
            timeout_ms,
        });
    }

    Ok(result)
}

/// Multi-model prediction (ensemble)
pub struct EnsemblePredictor<'a> {
    engines: Vec<&'a InferenceEngine>,
    weights: Vec<f64>,
}

impl<'a> EnsemblePredictor<'a> {
    /// Create new ensemble predictor
    pub fn new(engines: Vec<&'a InferenceEngine>) -> Result<Self> {
        if engines.is_empty() {
            return Err(ServingError::ValidationError {
                field: "engines".to_string(),
                message: "At least one engine required".to_string(),
            });
        }

        let n = engines.len();
        let weights = vec![1.0 / n as f64; n];

        Ok(Self { engines, weights })
    }

    /// Create ensemble predictor with custom weights
    pub fn with_weights(engines: Vec<&'a InferenceEngine>, weights: Vec<f64>) -> Result<Self> {
        if engines.is_empty() {
            return Err(ServingError::ValidationError {
                field: "engines".to_string(),
                message: "At least one engine required".to_string(),
            });
        }

        if engines.len() != weights.len() {
            return Err(ServingError::ValidationError {
                field: "weights".to_string(),
                message: "Number of weights must match number of engines".to_string(),
            });
        }

        // Normalize weights
        let sum: f64 = weights.iter().sum();
        if sum.abs() < 1e-10 {
            return Err(ServingError::ValidationError {
                field: "weights".to_string(),
                message: "Weights sum must be non-zero".to_string(),
            });
        }

        let normalized_weights: Vec<f64> = weights.iter().map(|w| w / sum).collect();

        Ok(Self {
            engines,
            weights: normalized_weights,
        })
    }

    /// Predict using weighted ensemble
    pub fn predict(&self, input: &Array<f64>) -> Result<Array<f64>> {
        let mut predictions = Vec::new();

        // Get predictions from all models
        for engine in &self.engines {
            let output = engine.infer(input)?;
            predictions.push(output);
        }

        // Weighted average
        let first_shape = predictions[0].shape();
        let size = predictions[0].size();

        let mut weighted_sum = vec![0.0; size];

        for (pred, &weight) in predictions.iter().zip(self.weights.iter()) {
            let data = pred.to_vec();
            for (i, &value) in data.iter().enumerate() {
                weighted_sum[i] += value * weight;
            }
        }

        let shape = first_shape.to_vec();
        Ok(Array::from_vec(weighted_sum).reshape(&shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_modules::serving::{InferenceEngine, Model};

    // Mock model for testing
    struct MockModel;

    impl Model for MockModel {
        fn forward(&self, input: &Array<f64>) -> Result<Array<f64>> {
            Ok(input.multiply_scalar(2.0))
        }

        fn name(&self) -> &str {
            "mock_model"
        }

        fn input_shape(&self) -> Vec<Option<usize>> {
            vec![None, Some(3)]
        }

        fn output_shape(&self) -> Vec<Option<usize>> {
            vec![None, Some(3)]
        }
    }

    #[test]
    fn test_prediction_request_creation() {
        let input = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let req = PredictionRequest::new(input);

        assert!(req.preprocess);
        assert!(req.request_id.is_none());
    }

    #[test]
    fn test_prediction_request_with_options() {
        let input = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let req = PredictionRequest::new(input)
            .with_id("test_id".to_string())
            .without_preprocessing()
            .with_timeout(1000);

        assert_eq!(req.request_id, Some("test_id".to_string()));
        assert!(!req.preprocess);
        assert_eq!(req.timeout_ms, Some(1000));
    }

    #[test]
    fn test_predict_sync() {
        let model = Box::new(MockModel);
        let engine = InferenceEngine::new(model);

        let input = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
        let response = predict_sync(&engine, &input, None).expect("Prediction should succeed");

        assert_eq!(response.output.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_predict_batch() {
        let model = Box::new(MockModel);
        let engine = InferenceEngine::new(model);

        let input1 = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
        let input2 = Array::from_vec(vec![4.0, 5.0, 6.0]).reshape(&[1, 3]);

        let response = predict_batch(&engine, &[input1, input2], None)
            .expect("Batch prediction should succeed");

        assert_eq!(response.outputs.len(), 2);
        assert_eq!(response.outputs[0].to_vec(), vec![2.0, 4.0, 6.0]);
        assert_eq!(response.outputs[1].to_vec(), vec![8.0, 10.0, 12.0]);
        assert!(response.throughput > 0.0);
    }

    #[test]
    fn test_streaming_predictor() {
        let model = Box::new(MockModel);
        let engine = InferenceEngine::new(model);

        let input1 = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
        let input2 = Array::from_vec(vec![4.0, 5.0, 6.0]).reshape(&[1, 3]);

        let mut predictor = StreamingPredictor::new(&engine, vec![input1, input2], None);

        assert_eq!(predictor.total(), 2);
        assert_eq!(predictor.position(), 0);
        assert!(!predictor.is_complete());

        let first = predictor.next().expect("Should have first prediction");
        assert!(first.is_ok());
        assert_eq!(predictor.position(), 1);

        let second = predictor.next().expect("Should have second prediction");
        assert!(second.is_ok());
        assert_eq!(predictor.position(), 2);

        assert!(predictor.is_complete());
        assert!(predictor.next().is_none());
    }

    #[test]
    fn test_batch_prediction_request() {
        let input1 = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let input2 = Array::from_vec(vec![4.0, 5.0, 6.0]);

        let req = BatchPredictionRequest::new(vec![input1, input2])
            .with_ids(vec!["id1".to_string(), "id2".to_string()])
            .with_timeout(5000);

        assert_eq!(req.inputs.len(), 2);
        assert_eq!(req.request_ids.as_ref().unwrap().len(), 2);
        assert_eq!(req.timeout_ms, Some(5000));
    }

    #[test]
    fn test_ensemble_predictor() {
        let model1 = Box::new(MockModel);
        let model2 = Box::new(MockModel);

        let engine1 = InferenceEngine::new(model1);
        let engine2 = InferenceEngine::new(model2);

        let ensemble = EnsemblePredictor::new(vec![&engine1, &engine2])
            .expect("Ensemble creation should succeed");

        let input = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
        let output = ensemble
            .predict(&input)
            .expect("Ensemble prediction should succeed");

        // Both models multiply by 2, weighted average should also be multiply by 2
        assert_eq!(output.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_ensemble_predictor_with_weights() {
        let model1 = Box::new(MockModel);
        let model2 = Box::new(MockModel);

        let engine1 = InferenceEngine::new(model1);
        let engine2 = InferenceEngine::new(model2);

        let ensemble = EnsemblePredictor::with_weights(vec![&engine1, &engine2], vec![0.3, 0.7])
            .expect("Ensemble creation should succeed");

        let input = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
        let output = ensemble
            .predict(&input)
            .expect("Ensemble prediction should succeed");

        // Weights sum to 1.0, both models multiply by 2
        let result = output.to_vec();
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10);
        assert!((result[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_ensemble_predictor_empty_engines() {
        let result = EnsemblePredictor::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ensemble_predictor_weight_mismatch() {
        let model = Box::new(MockModel);
        let engine = InferenceEngine::new(model);

        let result = EnsemblePredictor::with_weights(vec![&engine], vec![0.5, 0.5]);
        assert!(result.is_err());
    }
}
