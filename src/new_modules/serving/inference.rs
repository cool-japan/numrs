//! # Inference Engine
//!
//! Optimized inference engine for production ML serving with batch processing,
//! caching, and request queuing.

use super::{Result, ServingError};
use crate::array::Array;
use crate::error::NumRs2Error;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Model trait for inference
pub trait Model: Send + Sync {
    /// Perform forward pass
    fn forward(&self, input: &Array<f64>) -> Result<Array<f64>>;

    /// Get model name
    fn name(&self) -> &str;

    /// Get model input shape (None means any size for that dimension)
    fn input_shape(&self) -> Vec<Option<usize>>;

    /// Get model output shape (None means any size for that dimension)
    fn output_shape(&self) -> Vec<Option<usize>>;

    /// Warmup model (optional pre-execution for consistent latency)
    fn warmup(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Inference request with unique ID
#[derive(Clone)]
pub struct InferenceRequest {
    /// Unique request ID
    pub id: String,

    /// Input data
    pub input: Array<f64>,

    /// Request timestamp
    pub timestamp: Instant,

    /// Request priority (higher = more important)
    pub priority: i32,
}

impl InferenceRequest {
    /// Create new inference request
    pub fn new(id: String, input: Array<f64>) -> Self {
        Self {
            id,
            input,
            timestamp: Instant::now(),
            priority: 0,
        }
    }

    /// Create new inference request with priority
    pub fn with_priority(id: String, input: Array<f64>, priority: i32) -> Self {
        Self {
            id,
            input,
            timestamp: Instant::now(),
            priority,
        }
    }
}

/// Inference response
#[derive(Clone)]
pub struct InferenceResponse {
    /// Request ID
    pub id: String,

    /// Output data
    pub output: Array<f64>,

    /// Inference latency in milliseconds
    pub latency_ms: f64,
}

/// Batch inference configuration
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,

    /// Batch timeout in milliseconds
    pub timeout_ms: u64,

    /// Enable dynamic batching
    pub dynamic_batching: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            timeout_ms: 10,
            dynamic_batching: true,
        }
    }
}

/// Inference cache for memoization
pub struct InferenceCache {
    cache: RwLock<HashMap<Vec<u8>, Array<f64>>>,
    max_entries: usize,
    hits: Mutex<u64>,
    misses: Mutex<u64>,
}

impl InferenceCache {
    /// Create new inference cache
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_entries,
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        }
    }

    /// Get cached result
    pub fn get(&self, key: &[u8]) -> Option<Array<f64>> {
        let cache = self.cache.read().map_err(|_| ()).ok()?;
        if let Some(result) = cache.get(key) {
            if let Ok(mut hits) = self.hits.lock() {
                *hits += 1;
            }
            Some(result.clone())
        } else {
            if let Ok(mut misses) = self.misses.lock() {
                *misses += 1;
            }
            None
        }
    }

    /// Put result in cache
    pub fn put(&self, key: Vec<u8>, value: Array<f64>) -> Result<()> {
        let mut cache = self
            .cache
            .write()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire cache write lock".to_string(),
            })?;

        // Evict oldest entry if cache is full
        if cache.len() >= self.max_entries {
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }

        cache.insert(key, value);
        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.lock().map(|h| *h).unwrap_or(0);
        let misses = self.misses.lock().map(|m| *m).unwrap_or(0);
        let size = self.cache.read().map(|c| c.len()).unwrap_or(0);

        CacheStats {
            hits,
            misses,
            size,
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear cache
    pub fn clear(&self) -> Result<()> {
        let mut cache = self
            .cache
            .write()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire cache write lock".to_string(),
            })?;
        cache.clear();
        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub hit_rate: f64,
}

/// Request queue for batch processing
pub struct RequestQueue {
    queue: Mutex<VecDeque<InferenceRequest>>,
    max_size: usize,
}

impl RequestQueue {
    /// Create new request queue
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            max_size,
        }
    }

    /// Enqueue request
    pub fn enqueue(&self, request: InferenceRequest) -> Result<()> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire queue lock".to_string(),
            })?;

        if queue.len() >= self.max_size {
            return Err(ServingError::Other {
                message: format!("Request queue full (max: {})", self.max_size),
            });
        }

        // Insert based on priority (higher priority first)
        let insert_pos = queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(queue.len());

        queue.insert(insert_pos, request);
        Ok(())
    }

    /// Dequeue batch of requests
    pub fn dequeue_batch(&self, max_size: usize) -> Vec<InferenceRequest> {
        let mut queue = match self.queue.lock() {
            Ok(q) => q,
            Err(_) => return Vec::new(),
        };

        let batch_size = max_size.min(queue.len());
        queue.drain(0..batch_size).collect()
    }

    /// Get queue size
    pub fn size(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.lock().map(|q| q.is_empty()).unwrap_or(true)
    }
}

/// Inference engine for optimized model serving
pub struct InferenceEngine {
    model: Arc<RwLock<Box<dyn Model>>>,
    cache: Option<InferenceCache>,
    queue: RequestQueue,
    batch_config: BatchConfig,
    warmed_up: Mutex<bool>,
}

impl InferenceEngine {
    /// Create new inference engine
    pub fn new(model: Box<dyn Model>) -> Self {
        Self {
            model: Arc::new(RwLock::new(model)),
            cache: None,
            queue: RequestQueue::new(1000),
            batch_config: BatchConfig::default(),
            warmed_up: Mutex::new(false),
        }
    }

    /// Create inference engine with cache
    pub fn with_cache(model: Box<dyn Model>, cache_size: usize) -> Self {
        Self {
            model: Arc::new(RwLock::new(model)),
            cache: Some(InferenceCache::new(cache_size)),
            queue: RequestQueue::new(1000),
            batch_config: BatchConfig::default(),
            warmed_up: Mutex::new(false),
        }
    }

    /// Create inference engine with custom configuration
    pub fn with_config(
        model: Box<dyn Model>,
        batch_config: BatchConfig,
        cache_size: Option<usize>,
    ) -> Self {
        Self {
            model: Arc::new(RwLock::new(model)),
            cache: cache_size.map(InferenceCache::new),
            queue: RequestQueue::new(1000),
            batch_config,
            warmed_up: Mutex::new(false),
        }
    }

    /// Warmup model for consistent latency
    pub fn warmup(&self) -> Result<()> {
        let mut model = self
            .model
            .write()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire model write lock".to_string(),
            })?;

        model.warmup()?;

        let mut warmed_up = self
            .warmed_up
            .lock()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire warmup lock".to_string(),
            })?;
        *warmed_up = true;

        Ok(())
    }

    /// Perform synchronous inference
    pub fn infer(&self, input: &Array<f64>) -> Result<Array<f64>> {
        let start = Instant::now();

        // Check cache if enabled
        if let Some(ref cache) = self.cache {
            let key = self.compute_cache_key(input)?;
            if let Some(cached_result) = cache.get(&key) {
                return Ok(cached_result);
            }
        }

        // Perform inference
        let model = self
            .model
            .read()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire model read lock".to_string(),
            })?;

        let output = model.forward(input)?;

        // Update cache if enabled
        if let Some(ref cache) = self.cache {
            let key = self.compute_cache_key(input)?;
            let _ = cache.put(key, output.clone());
        }

        Ok(output)
    }

    /// Perform batch inference
    pub fn infer_batch(&self, inputs: &[Array<f64>]) -> Result<Vec<Array<f64>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let model = self
            .model
            .read()
            .map_err(|_| ServingError::ConcurrencyError {
                message: "Failed to acquire model read lock".to_string(),
            })?;

        let mut outputs = Vec::with_capacity(inputs.len());

        for input in inputs {
            let output = model.forward(input)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Enqueue request for batch processing
    pub fn enqueue(&self, request: InferenceRequest) -> Result<()> {
        self.queue.enqueue(request)
    }

    /// Process queued requests in batch
    pub fn process_queue(&self) -> Result<Vec<InferenceResponse>> {
        let requests = self.queue.dequeue_batch(self.batch_config.max_batch_size);

        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let inputs: Vec<Array<f64>> = requests.iter().map(|r| r.input.clone()).collect();

        let outputs = self.infer_batch(&inputs)?;

        let responses: Vec<InferenceResponse> = requests
            .into_iter()
            .zip(outputs)
            .map(|(req, output)| {
                let latency_ms = req.timestamp.elapsed().as_secs_f64() * 1000.0;
                InferenceResponse {
                    id: req.id,
                    output,
                    latency_ms,
                }
            })
            .collect();

        Ok(responses)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }

    /// Clear cache
    pub fn clear_cache(&self) -> Result<()> {
        if let Some(ref cache) = self.cache {
            cache.clear()?;
        }
        Ok(())
    }

    /// Get queue size
    pub fn queue_size(&self) -> usize {
        self.queue.size()
    }

    /// Compute cache key from input
    fn compute_cache_key(&self, input: &Array<f64>) -> Result<Vec<u8>> {
        // Simple hash based on input values
        let data = input.to_vec();
        let mut key = Vec::with_capacity(data.len() * 8);

        for &value in &data {
            key.extend_from_slice(&value.to_ne_bytes());
        }

        Ok(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock model for testing
    struct MockModel {
        name: String,
        multiplier: f64,
    }

    impl MockModel {
        fn new(multiplier: f64) -> Self {
            Self {
                name: "mock_model".to_string(),
                multiplier,
            }
        }
    }

    impl Model for MockModel {
        fn forward(&self, input: &Array<f64>) -> Result<Array<f64>> {
            Ok(input.multiply_scalar(self.multiplier))
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn input_shape(&self) -> Vec<Option<usize>> {
            vec![None, Some(3)]
        }

        fn output_shape(&self) -> Vec<Option<usize>> {
            vec![None, Some(3)]
        }
    }

    #[test]
    fn test_inference_engine_creation() {
        let model = Box::new(MockModel::new(2.0));
        let engine = InferenceEngine::new(model);
        assert!(engine.cache.is_none());
    }

    #[test]
    fn test_inference_engine_with_cache() {
        let model = Box::new(MockModel::new(2.0));
        let engine = InferenceEngine::with_cache(model, 100);
        assert!(engine.cache.is_some());
    }

    #[test]
    fn test_synchronous_inference() {
        let model = Box::new(MockModel::new(2.0));
        let engine = InferenceEngine::new(model);

        let input = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
        let output = engine.infer(&input).expect("Inference should succeed");

        assert_eq!(output.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_batch_inference() {
        let model = Box::new(MockModel::new(2.0));
        let engine = InferenceEngine::new(model);

        let input1 = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);
        let input2 = Array::from_vec(vec![4.0, 5.0, 6.0]).reshape(&[1, 3]);

        let outputs = engine
            .infer_batch(&[input1, input2])
            .expect("Batch inference should succeed");

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].to_vec(), vec![2.0, 4.0, 6.0]);
        assert_eq!(outputs[1].to_vec(), vec![8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_inference_with_cache() {
        let model = Box::new(MockModel::new(2.0));
        let engine = InferenceEngine::with_cache(model, 10);

        let input = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]);

        // First call - cache miss
        let output1 = engine
            .infer(&input)
            .expect("First inference should succeed");
        let stats1 = engine
            .cache_stats()
            .expect("Cache stats should be available");
        assert_eq!(stats1.misses, 1);

        // Second call - cache hit
        let output2 = engine
            .infer(&input)
            .expect("Second inference should succeed");
        let stats2 = engine
            .cache_stats()
            .expect("Cache stats should be available");
        assert_eq!(stats2.hits, 1);

        assert_eq!(output1.to_vec(), output2.to_vec());
    }

    #[test]
    fn test_request_queue_enqueue_dequeue() {
        let queue = RequestQueue::new(100);

        let req1 = InferenceRequest::new("req1".to_string(), Array::from_vec(vec![1.0, 2.0, 3.0]));
        let req2 = InferenceRequest::new("req2".to_string(), Array::from_vec(vec![4.0, 5.0, 6.0]));

        queue.enqueue(req1).expect("Enqueue should succeed");
        queue.enqueue(req2).expect("Enqueue should succeed");

        assert_eq!(queue.size(), 2);

        let batch = queue.dequeue_batch(2);
        assert_eq!(batch.len(), 2);
        assert_eq!(queue.size(), 0);
    }

    #[test]
    fn test_request_queue_priority() {
        let queue = RequestQueue::new(100);

        let req_low =
            InferenceRequest::with_priority("low".to_string(), Array::from_vec(vec![1.0]), 1);
        let req_high =
            InferenceRequest::with_priority("high".to_string(), Array::from_vec(vec![2.0]), 10);

        queue.enqueue(req_low).expect("Enqueue should succeed");
        queue.enqueue(req_high).expect("Enqueue should succeed");

        let batch = queue.dequeue_batch(1);
        assert_eq!(batch[0].id, "high");
    }

    #[test]
    fn test_cache_eviction() {
        let cache = InferenceCache::new(2);

        let key1 = vec![1, 2, 3];
        let key2 = vec![4, 5, 6];
        let key3 = vec![7, 8, 9];

        let value = Array::from_vec(vec![1.0, 2.0, 3.0]);

        cache
            .put(key1.clone(), value.clone())
            .expect("Put should succeed");
        cache
            .put(key2.clone(), value.clone())
            .expect("Put should succeed");
        cache
            .put(key3.clone(), value.clone())
            .expect("Put should succeed");

        let stats = cache.stats();
        assert_eq!(stats.size, 2); // Only 2 entries should remain
    }

    #[test]
    fn test_inference_request_creation() {
        let input = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let req = InferenceRequest::new("test".to_string(), input);

        assert_eq!(req.id, "test");
        assert_eq!(req.priority, 0);
    }

    #[test]
    fn test_inference_request_with_priority() {
        let input = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let req = InferenceRequest::with_priority("test".to_string(), input, 5);

        assert_eq!(req.id, "test");
        assert_eq!(req.priority, 5);
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.timeout_ms, 10);
        assert!(config.dynamic_batching);
    }

    #[test]
    fn test_cache_clear() {
        let cache = InferenceCache::new(10);

        let key = vec![1, 2, 3];
        let value = Array::from_vec(vec![1.0, 2.0, 3.0]);

        cache
            .put(key.clone(), value.clone())
            .expect("Put should succeed");
        assert_eq!(cache.stats().size, 1);

        cache.clear().expect("Clear should succeed");
        assert_eq!(cache.stats().size, 0);
    }

    #[test]
    fn test_engine_warmup() {
        let model = Box::new(MockModel::new(2.0));
        let engine = InferenceEngine::new(model);

        engine.warmup().expect("Warmup should succeed");

        let warmed_up = engine.warmed_up.lock().expect("Lock should succeed");
        assert!(*warmed_up);
    }
}
