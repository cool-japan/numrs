//! # Serving Metrics
//!
//! Request latency tracking, throughput monitoring, accuracy metrics, and resource utilization.

use super::{Result, ServingError};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Latency percentiles
#[derive(Debug, Clone)]
pub struct LatencyPercentiles {
    /// 50th percentile (median)
    pub p50: f64,

    /// 95th percentile
    pub p95: f64,

    /// 99th percentile
    pub p99: f64,

    /// Mean latency
    pub mean: f64,

    /// Minimum latency
    pub min: f64,

    /// Maximum latency
    pub max: f64,
}

/// Latency tracker
pub struct LatencyTracker {
    start_time: Instant,
}

impl LatencyTracker {
    /// Start tracking latency
    pub fn start() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64() * 1000.0
    }

    /// Get elapsed duration
    pub fn elapsed_duration(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Latency histogram
pub struct LatencyHistogram {
    latencies: Mutex<VecDeque<f64>>,
    max_samples: usize,
}

impl LatencyHistogram {
    /// Create new latency histogram
    pub fn new(max_samples: usize) -> Self {
        Self {
            latencies: Mutex::new(VecDeque::new()),
            max_samples,
        }
    }

    /// Record latency
    pub fn record(&self, latency_ms: f64) -> Result<()> {
        let mut latencies = self
            .latencies
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire latencies lock".to_string(),
            })?;

        if latencies.len() >= self.max_samples {
            latencies.pop_front();
        }

        latencies.push_back(latency_ms);
        Ok(())
    }

    /// Get latency percentiles
    pub fn percentiles(&self) -> Result<LatencyPercentiles> {
        let latencies = self
            .latencies
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire latencies lock".to_string(),
            })?;

        if latencies.is_empty() {
            return Ok(LatencyPercentiles {
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                mean: 0.0,
                min: 0.0,
                max: 0.0,
            });
        }

        let mut sorted: Vec<f64> = latencies.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = sorted.len();
        let p50_idx = (len as f64 * 0.50) as usize;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;

        let p50 = sorted.get(p50_idx.min(len - 1)).cloned().unwrap_or(0.0);
        let p95 = sorted.get(p95_idx.min(len - 1)).cloned().unwrap_or(0.0);
        let p99 = sorted.get(p99_idx.min(len - 1)).cloned().unwrap_or(0.0);

        let mean = sorted.iter().sum::<f64>() / len as f64;
        let min = sorted.first().cloned().unwrap_or(0.0);
        let max = sorted.last().cloned().unwrap_or(0.0);

        Ok(LatencyPercentiles {
            p50,
            p95,
            p99,
            mean,
            min,
            max,
        })
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.latencies.lock().map(|l| l.len()).unwrap_or(0)
    }

    /// Clear histogram
    pub fn clear(&self) -> Result<()> {
        let mut latencies = self
            .latencies
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire latencies lock".to_string(),
            })?;
        latencies.clear();
        Ok(())
    }
}

/// Throughput tracker
pub struct ThroughputTracker {
    requests: Mutex<VecDeque<Instant>>,
    window_size: Duration,
}

impl ThroughputTracker {
    /// Create new throughput tracker
    pub fn new(window_size: Duration) -> Self {
        Self {
            requests: Mutex::new(VecDeque::new()),
            window_size,
        }
    }

    /// Record request
    pub fn record(&self) -> Result<()> {
        let now = Instant::now();

        let mut requests = self
            .requests
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire requests lock".to_string(),
            })?;

        // Remove old requests outside window
        let cutoff = now - self.window_size;
        while let Some(&first) = requests.front() {
            if first < cutoff {
                requests.pop_front();
            } else {
                break;
            }
        }

        requests.push_back(now);
        Ok(())
    }

    /// Get current throughput (requests per second)
    pub fn throughput(&self) -> Result<f64> {
        let requests = self
            .requests
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire requests lock".to_string(),
            })?;

        if requests.is_empty() {
            return Ok(0.0);
        }

        let window_secs = self.window_size.as_secs_f64();
        Ok(requests.len() as f64 / window_secs)
    }

    /// Get request count in window
    pub fn request_count(&self) -> usize {
        self.requests.lock().map(|r| r.len()).unwrap_or(0)
    }

    /// Clear tracker
    pub fn clear(&self) -> Result<()> {
        let mut requests = self
            .requests
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire requests lock".to_string(),
            })?;
        requests.clear();
        Ok(())
    }
}

/// Accuracy metrics
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    /// Number of correct predictions
    pub correct: usize,

    /// Total number of predictions
    pub total: usize,

    /// Accuracy (correct / total)
    pub accuracy: f64,

    /// Mean absolute error (for regression)
    pub mae: Option<f64>,

    /// Mean squared error (for regression)
    pub mse: Option<f64>,
}

impl AccuracyMetrics {
    /// Create new accuracy metrics
    pub fn new() -> Self {
        Self {
            correct: 0,
            total: 0,
            accuracy: 0.0,
            mae: None,
            mse: None,
        }
    }

    /// Update with classification result
    pub fn update_classification(&mut self, predicted: usize, actual: usize) {
        self.total += 1;
        if predicted == actual {
            self.correct += 1;
        }
        self.accuracy = self.correct as f64 / self.total as f64;
    }

    /// Update with regression result
    pub fn update_regression(&mut self, predicted: f64, actual: f64) {
        let error = (predicted - actual).abs();
        let squared_error = (predicted - actual).powi(2);

        let mae = self.mae.unwrap_or(0.0);
        let mse = self.mse.unwrap_or(0.0);

        self.total += 1;

        // Incremental mean update
        self.mae = Some(mae + (error - mae) / self.total as f64);
        self.mse = Some(mse + (squared_error - mse) / self.total as f64);
    }

    /// Get root mean squared error
    pub fn rmse(&self) -> Option<f64> {
        self.mse.map(|mse| mse.sqrt())
    }
}

impl Default for AccuracyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// CPU utilization percentage (0-100)
    pub cpu_percent: f64,

    /// Memory usage in bytes
    pub memory_bytes: usize,

    /// Memory utilization percentage (0-100)
    pub memory_percent: f64,

    /// Number of active requests
    pub active_requests: usize,

    /// Queue depth
    pub queue_depth: usize,
}

impl ResourceMetrics {
    /// Create new resource metrics
    pub fn new() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_bytes: 0,
            memory_percent: 0.0,
            active_requests: 0,
            queue_depth: 0,
        }
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive serving metrics
pub struct ServingMetrics {
    latency_histogram: LatencyHistogram,
    throughput_tracker: ThroughputTracker,
    accuracy_metrics: Mutex<AccuracyMetrics>,
    resource_metrics: Mutex<ResourceMetrics>,
    error_count: Mutex<usize>,
    success_count: Mutex<usize>,
}

impl ServingMetrics {
    /// Create new serving metrics
    pub fn new() -> Self {
        Self {
            latency_histogram: LatencyHistogram::new(10000),
            throughput_tracker: ThroughputTracker::new(Duration::from_secs(60)),
            accuracy_metrics: Mutex::new(AccuracyMetrics::new()),
            resource_metrics: Mutex::new(ResourceMetrics::new()),
            error_count: Mutex::new(0),
            success_count: Mutex::new(0),
        }
    }

    /// Record successful prediction
    pub fn record_success(&self, latency_ms: f64) -> Result<()> {
        self.latency_histogram.record(latency_ms)?;
        self.throughput_tracker.record()?;

        let mut success_count =
            self.success_count
                .lock()
                .map_err(|_| ServingError::MetricsError {
                    message: "Failed to acquire success_count lock".to_string(),
                })?;
        *success_count += 1;

        Ok(())
    }

    /// Record error
    pub fn record_error(&self) -> Result<()> {
        let mut error_count = self
            .error_count
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire error_count lock".to_string(),
            })?;
        *error_count += 1;

        Ok(())
    }

    /// Get latency percentiles
    pub fn latency_percentiles(&self) -> Result<LatencyPercentiles> {
        self.latency_histogram.percentiles()
    }

    /// Get current throughput
    pub fn throughput(&self) -> Result<f64> {
        self.throughput_tracker.throughput()
    }

    /// Update accuracy metrics (classification)
    pub fn update_classification_accuracy(&self, predicted: usize, actual: usize) -> Result<()> {
        let mut accuracy =
            self.accuracy_metrics
                .lock()
                .map_err(|_| ServingError::MetricsError {
                    message: "Failed to acquire accuracy_metrics lock".to_string(),
                })?;
        accuracy.update_classification(predicted, actual);
        Ok(())
    }

    /// Update accuracy metrics (regression)
    pub fn update_regression_accuracy(&self, predicted: f64, actual: f64) -> Result<()> {
        let mut accuracy =
            self.accuracy_metrics
                .lock()
                .map_err(|_| ServingError::MetricsError {
                    message: "Failed to acquire accuracy_metrics lock".to_string(),
                })?;
        accuracy.update_regression(predicted, actual);
        Ok(())
    }

    /// Get accuracy metrics
    pub fn accuracy_metrics(&self) -> Result<AccuracyMetrics> {
        let accuracy = self
            .accuracy_metrics
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire accuracy_metrics lock".to_string(),
            })?;
        Ok(accuracy.clone())
    }

    /// Update resource metrics
    pub fn update_resource_metrics(&self, metrics: ResourceMetrics) -> Result<()> {
        let mut resource_metrics =
            self.resource_metrics
                .lock()
                .map_err(|_| ServingError::MetricsError {
                    message: "Failed to acquire resource_metrics lock".to_string(),
                })?;
        *resource_metrics = metrics;
        Ok(())
    }

    /// Get resource metrics
    pub fn resource_metrics(&self) -> Result<ResourceMetrics> {
        let metrics = self
            .resource_metrics
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire resource_metrics lock".to_string(),
            })?;
        Ok(metrics.clone())
    }

    /// Get error rate
    pub fn error_rate(&self) -> Result<f64> {
        let errors = self
            .error_count
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire error_count lock".to_string(),
            })?;

        let successes = self
            .success_count
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire success_count lock".to_string(),
            })?;

        let total = *errors + *successes;
        if total == 0 {
            return Ok(0.0);
        }

        Ok(*errors as f64 / total as f64)
    }

    /// Get success count
    pub fn success_count(&self) -> usize {
        self.success_count.lock().map(|c| *c).unwrap_or(0)
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.error_count.lock().map(|c| *c).unwrap_or(0)
    }

    /// Reset all metrics
    pub fn reset(&self) -> Result<()> {
        self.latency_histogram.clear()?;
        self.throughput_tracker.clear()?;

        let mut accuracy =
            self.accuracy_metrics
                .lock()
                .map_err(|_| ServingError::MetricsError {
                    message: "Failed to acquire accuracy_metrics lock".to_string(),
                })?;
        *accuracy = AccuracyMetrics::new();

        let mut errors = self
            .error_count
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire error_count lock".to_string(),
            })?;
        *errors = 0;

        let mut successes = self
            .success_count
            .lock()
            .map_err(|_| ServingError::MetricsError {
                message: "Failed to acquire success_count lock".to_string(),
            })?;
        *successes = 0;

        Ok(())
    }
}

impl Default for ServingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_latency_tracker() {
        let tracker = LatencyTracker::start();
        thread::sleep(Duration::from_millis(10));
        let elapsed = tracker.elapsed();
        assert!(elapsed >= 10.0);
    }

    #[test]
    fn test_latency_histogram() {
        let histogram = LatencyHistogram::new(100);

        histogram.record(10.0).expect("Record should succeed");
        histogram.record(20.0).expect("Record should succeed");
        histogram.record(30.0).expect("Record should succeed");

        let percentiles = histogram.percentiles().expect("Percentiles should succeed");

        assert!(percentiles.min > 0.0);
        assert!(percentiles.max > 0.0);
        assert!(percentiles.mean > 0.0);
    }

    #[test]
    fn test_latency_histogram_percentiles() {
        let histogram = LatencyHistogram::new(100);

        for i in 1..=100 {
            histogram.record(i as f64).expect("Record should succeed");
        }

        let percentiles = histogram.percentiles().expect("Percentiles should succeed");

        assert!((percentiles.p50 - 50.0).abs() < 5.0);
        assert!((percentiles.p95 - 95.0).abs() < 5.0);
        assert!((percentiles.p99 - 99.0).abs() < 5.0);
    }

    #[test]
    fn test_throughput_tracker() {
        let tracker = ThroughputTracker::new(Duration::from_secs(1));

        tracker.record().expect("Record should succeed");
        tracker.record().expect("Record should succeed");
        tracker.record().expect("Record should succeed");

        let throughput = tracker.throughput().expect("Throughput should succeed");
        assert!(throughput > 0.0);
        assert_eq!(tracker.request_count(), 3);
    }

    #[test]
    fn test_accuracy_metrics_classification() {
        let mut metrics = AccuracyMetrics::new();

        metrics.update_classification(0, 0); // Correct
        metrics.update_classification(1, 0); // Incorrect
        metrics.update_classification(0, 0); // Correct

        assert_eq!(metrics.total, 3);
        assert_eq!(metrics.correct, 2);
        assert!((metrics.accuracy - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_metrics_regression() {
        let mut metrics = AccuracyMetrics::new();

        metrics.update_regression(1.0, 1.5); // Error: 0.5
        metrics.update_regression(2.0, 2.5); // Error: 0.5

        assert_eq!(metrics.total, 2);
        assert!(metrics.mae.is_some());
        assert!((metrics.mae.expect("test: MAE is some") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_resource_metrics() {
        let metrics = ResourceMetrics::new();

        assert_eq!(metrics.cpu_percent, 0.0);
        assert_eq!(metrics.memory_bytes, 0);
        assert_eq!(metrics.active_requests, 0);
    }

    #[test]
    fn test_serving_metrics() {
        let metrics = ServingMetrics::new();

        metrics
            .record_success(10.0)
            .expect("Record success should succeed");
        metrics
            .record_success(20.0)
            .expect("Record success should succeed");
        metrics.record_error().expect("Record error should succeed");

        assert_eq!(metrics.success_count(), 2);
        assert_eq!(metrics.error_count(), 1);

        let error_rate = metrics.error_rate().expect("Error rate should succeed");
        assert!((error_rate - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_serving_metrics_latency() {
        let metrics = ServingMetrics::new();

        metrics.record_success(10.0).expect("Record should succeed");
        metrics.record_success(20.0).expect("Record should succeed");
        metrics.record_success(30.0).expect("Record should succeed");

        let percentiles = metrics
            .latency_percentiles()
            .expect("Latency percentiles should succeed");

        assert!(percentiles.mean > 0.0);
    }

    #[test]
    fn test_serving_metrics_throughput() {
        let metrics = ServingMetrics::new();

        metrics.record_success(10.0).expect("Record should succeed");
        metrics.record_success(20.0).expect("Record should succeed");

        let throughput = metrics.throughput().expect("Throughput should succeed");
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_serving_metrics_accuracy() {
        let metrics = ServingMetrics::new();

        metrics
            .update_classification_accuracy(0, 0)
            .expect("Update should succeed");
        metrics
            .update_classification_accuracy(1, 1)
            .expect("Update should succeed");

        let accuracy = metrics
            .accuracy_metrics()
            .expect("Get accuracy should succeed");

        assert_eq!(accuracy.correct, 2);
        assert_eq!(accuracy.total, 2);
        assert_eq!(accuracy.accuracy, 1.0);
    }

    #[test]
    fn test_serving_metrics_reset() {
        let metrics = ServingMetrics::new();

        metrics.record_success(10.0).expect("Record should succeed");
        metrics.record_error().expect("Record should succeed");

        assert_eq!(metrics.success_count(), 1);
        assert_eq!(metrics.error_count(), 1);

        metrics.reset().expect("Reset should succeed");

        assert_eq!(metrics.success_count(), 0);
        assert_eq!(metrics.error_count(), 0);
    }
}
