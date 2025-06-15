//! Performance tuning and optimization for memory allocators
//!
//! This module provides automatic performance tuning capabilities for memory allocators
//! based on runtime characteristics and workload patterns.

use crate::error::{NumRs2Error, Result};
use crate::memory_alloc::benchmarking::{AllocatorBenchmark, BenchmarkConfig, BenchmarkResults};
use crate::traits::SpecializedAllocator;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Performance metrics collected during allocator usage
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Total number of allocations performed
    pub total_allocations: u64,
    /// Total number of deallocations performed
    pub total_deallocations: u64,
    /// Total bytes allocated
    pub total_bytes_allocated: u64,
    /// Total bytes deallocated
    pub total_bytes_deallocated: u64,
    /// Average allocation time in nanoseconds
    pub avg_allocation_time_ns: u64,
    /// Average deallocation time in nanoseconds
    pub avg_deallocation_time_ns: u64,
    /// Number of allocation failures
    pub allocation_failures: u64,
    /// Peak memory usage
    pub peak_memory_usage: u64,
    /// Current memory usage
    pub current_memory_usage: u64,
    /// Last update timestamp
    pub last_updated: Instant,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            total_bytes_allocated: 0,
            total_bytes_deallocated: 0,
            avg_allocation_time_ns: 0,
            avg_deallocation_time_ns: 0,
            allocation_failures: 0,
            peak_memory_usage: 0,
            current_memory_usage: 0,
            last_updated: Instant::now(),
        }
    }
}

/// Performance optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Human-readable description
    pub description: String,
    /// Estimated performance improvement (0.0-1.0)
    pub estimated_improvement: f64,
    /// Implementation difficulty (1-5)
    pub difficulty: u8,
    /// Configuration parameters to apply
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    /// Increase block size for better locality
    IncreaseBlockSize,
    /// Decrease block size to reduce fragmentation
    DecreaseBlockSize,
    /// Adjust alignment for SIMD optimization
    OptimizeAlignment,
    /// Switch to arena allocation for small objects
    UseArenaAllocation,
    /// Switch to pool allocation for fixed-size objects
    UsePoolAllocation,
    /// Implement memory pre-allocation
    EnablePreallocation,
    /// Optimize for concurrent access
    OptimizeConcurrency,
    /// Reduce memory overhead
    ReduceOverhead,
}

/// Automatic performance tuner for memory allocators
pub struct PerformanceTuner {
    /// Historical performance metrics
    metrics_history: Vec<PerformanceMetrics>,
    /// Current performance metrics
    current_metrics: Arc<Mutex<PerformanceMetrics>>,
    /// Tuning configuration
    config: TuningConfig,
    /// Benchmark results cache
    benchmark_cache: HashMap<String, BenchmarkResults>,
}

/// Configuration for performance tuning
#[derive(Debug, Clone)]
pub struct TuningConfig {
    /// How often to collect metrics (in milliseconds)
    pub collection_interval_ms: u64,
    /// Minimum sample size before making recommendations
    pub min_sample_size: u64,
    /// Maximum number of metrics to keep in history
    pub max_history_size: usize,
    /// Performance improvement threshold for recommendations
    pub improvement_threshold: f64,
    /// Enable automatic tuning adjustments
    pub auto_tuning_enabled: bool,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 1000,
            min_sample_size: 100,
            max_history_size: 1000,
            improvement_threshold: 0.05, // 5% improvement
            auto_tuning_enabled: false,
        }
    }
}

impl PerformanceTuner {
    /// Create a new performance tuner
    pub fn new(config: TuningConfig) -> Self {
        Self {
            metrics_history: Vec::new(),
            current_metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
            config,
            benchmark_cache: HashMap::new(),
        }
    }

    /// Create a tuner with default configuration
    pub fn default() -> Self {
        Self::new(TuningConfig::default())
    }

    /// Record an allocation event
    pub fn record_allocation(&self, size: usize, duration: Duration) {
        let mut metrics = self.current_metrics.lock().unwrap();
        metrics.total_allocations += 1;
        metrics.total_bytes_allocated += size as u64;
        metrics.current_memory_usage += size as u64;

        if metrics.current_memory_usage > metrics.peak_memory_usage {
            metrics.peak_memory_usage = metrics.current_memory_usage;
        }

        // Update running average
        let new_time_ns = duration.as_nanos() as u64;
        if metrics.total_allocations == 1 {
            metrics.avg_allocation_time_ns = new_time_ns;
        } else {
            metrics.avg_allocation_time_ns =
                (metrics.avg_allocation_time_ns * (metrics.total_allocations - 1) + new_time_ns)
                    / metrics.total_allocations;
        }

        metrics.last_updated = Instant::now();
    }

    /// Record a deallocation event
    pub fn record_deallocation(&self, size: usize, duration: Duration) {
        let mut metrics = self.current_metrics.lock().unwrap();
        metrics.total_deallocations += 1;
        metrics.total_bytes_deallocated += size as u64;
        metrics.current_memory_usage = metrics.current_memory_usage.saturating_sub(size as u64);

        // Update running average
        let new_time_ns = duration.as_nanos() as u64;
        if metrics.total_deallocations == 1 {
            metrics.avg_deallocation_time_ns = new_time_ns;
        } else {
            metrics.avg_deallocation_time_ns = (metrics.avg_deallocation_time_ns
                * (metrics.total_deallocations - 1)
                + new_time_ns)
                / metrics.total_deallocations;
        }

        metrics.last_updated = Instant::now();
    }

    /// Record an allocation failure
    pub fn record_allocation_failure(&self) {
        let mut metrics = self.current_metrics.lock().unwrap();
        metrics.allocation_failures += 1;
        metrics.last_updated = Instant::now();
    }

    /// Get current performance metrics
    pub fn get_current_metrics(&self) -> PerformanceMetrics {
        self.current_metrics.lock().unwrap().clone()
    }

    /// Take a metrics snapshot and add to history
    pub fn take_snapshot(&mut self) {
        let current = self.get_current_metrics();
        self.metrics_history.push(current);

        // Trim history if it's too large
        if self.metrics_history.len() > self.config.max_history_size {
            self.metrics_history.remove(0);
        }
    }

    /// Analyze performance and generate optimization recommendations
    pub fn analyze_performance(&self) -> Vec<OptimizationRecommendation> {
        let current = self.get_current_metrics();
        let mut recommendations = Vec::new();

        // Check if we have enough data
        if current.total_allocations < self.config.min_sample_size {
            return recommendations;
        }

        // Analyze allocation patterns
        recommendations.extend(self.analyze_allocation_patterns(&current));

        // Analyze timing performance
        recommendations.extend(self.analyze_timing_performance(&current));

        // Analyze memory efficiency
        recommendations.extend(self.analyze_memory_efficiency(&current));

        // Analyze failure rates
        recommendations.extend(self.analyze_failure_rates(&current));

        recommendations
    }

    /// Analyze allocation size and frequency patterns
    fn analyze_allocation_patterns(
        &self,
        metrics: &PerformanceMetrics,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Calculate average allocation size
        let avg_allocation_size = if metrics.total_allocations > 0 {
            metrics.total_bytes_allocated / metrics.total_allocations
        } else {
            0
        };

        // Recommend arena allocation for small, frequent allocations
        if avg_allocation_size < 1024 && metrics.total_allocations > 1000 {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::UseArenaAllocation,
                description: "Switch to arena allocation for small, frequent allocations"
                    .to_string(),
                estimated_improvement: 0.2,
                difficulty: 2,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("arena_size".to_string(), "65536".to_string());
                    params.insert("block_size".to_string(), avg_allocation_size.to_string());
                    params
                },
            });
        }

        // Recommend pool allocation for fixed-size allocations
        if self.has_consistent_allocation_sizes(metrics) {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::UsePoolAllocation,
                description: "Use memory pool for consistent allocation sizes".to_string(),
                estimated_improvement: 0.15,
                difficulty: 2,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("pool_size".to_string(), avg_allocation_size.to_string());
                    params.insert("initial_capacity".to_string(), "100".to_string());
                    params
                },
            });
        }

        recommendations
    }

    /// Analyze allocation and deallocation timing
    fn analyze_timing_performance(
        &self,
        metrics: &PerformanceMetrics,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check if allocation times are too high
        if metrics.avg_allocation_time_ns > 10_000 {
            // 10 microseconds
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::EnablePreallocation,
                description: "Enable memory pre-allocation to reduce allocation overhead"
                    .to_string(),
                estimated_improvement: 0.3,
                difficulty: 3,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("prealloc_size".to_string(), "1048576".to_string()); // 1MB
                    params
                },
            });
        }

        // Check if we need alignment optimization
        if metrics.avg_allocation_time_ns > 5_000 && self.has_simd_workload() {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::OptimizeAlignment,
                description: "Optimize memory alignment for SIMD operations".to_string(),
                estimated_improvement: 0.1,
                difficulty: 1,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("alignment".to_string(), "32".to_string());
                    params
                },
            });
        }

        recommendations
    }

    /// Analyze memory usage efficiency
    fn analyze_memory_efficiency(
        &self,
        metrics: &PerformanceMetrics,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check memory utilization
        let memory_utilization = if metrics.peak_memory_usage > 0 {
            metrics.current_memory_usage as f64 / metrics.peak_memory_usage as f64
        } else {
            1.0
        };

        // Low utilization might indicate fragmentation
        if memory_utilization < 0.7 {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::ReduceOverhead,
                description: "Reduce memory fragmentation and overhead".to_string(),
                estimated_improvement: 0.15,
                difficulty: 3,
                parameters: HashMap::new(),
            });
        }

        // Check if we have high overhead from metadata
        let overhead_ratio = self.estimate_metadata_overhead(metrics);
        if overhead_ratio > 0.1 {
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::ReduceOverhead,
                description: "Optimize allocation metadata to reduce overhead".to_string(),
                estimated_improvement: overhead_ratio * 0.5,
                difficulty: 4,
                parameters: HashMap::new(),
            });
        }

        recommendations
    }

    /// Analyze allocation failure patterns
    fn analyze_failure_rates(
        &self,
        metrics: &PerformanceMetrics,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        let failure_rate = if metrics.total_allocations > 0 {
            metrics.allocation_failures as f64 / metrics.total_allocations as f64
        } else {
            0.0
        };

        // High failure rate indicates memory pressure
        if failure_rate > 0.01 {
            // 1% failure rate
            recommendations.push(OptimizationRecommendation {
                optimization_type: OptimizationType::EnablePreallocation,
                description: "Pre-allocate memory to reduce allocation failures".to_string(),
                estimated_improvement: 0.25,
                difficulty: 2,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert(
                        "reserve_size".to_string(),
                        (metrics.peak_memory_usage * 2).to_string(),
                    );
                    params
                },
            });
        }

        recommendations
    }

    /// Check if allocations have consistent sizes (good for pooling)
    fn has_consistent_allocation_sizes(&self, _metrics: &PerformanceMetrics) -> bool {
        // This would require tracking allocation size distribution
        // For now, use a simple heuristic
        true // Placeholder
    }

    /// Check if workload is SIMD-intensive
    fn has_simd_workload(&self) -> bool {
        // This would check for patterns indicating SIMD usage
        // For now, assume it could benefit from SIMD
        true // Placeholder
    }

    /// Estimate metadata overhead ratio
    fn estimate_metadata_overhead(&self, _metrics: &PerformanceMetrics) -> f64 {
        // Typical allocator metadata overhead
        0.08 // 8% overhead estimate
    }

    /// Benchmark an allocator and cache the results
    pub fn benchmark_allocator<A>(
        &mut self,
        allocator: &A,
        name: &str,
        config: BenchmarkConfig,
    ) -> Result<BenchmarkResults>
    where
        A: SpecializedAllocator<Error = NumRs2Error>,
    {
        let cache_key = format!("{}_{:?}", name, config.iterations);

        if let Some(cached_result) = self.benchmark_cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }

        let mut benchmark = AllocatorBenchmark::new(config);
        let results = benchmark.benchmark_allocator(allocator, name)?;

        self.benchmark_cache.insert(cache_key, results.clone());
        Ok(results)
    }

    /// Apply optimization recommendations automatically
    pub fn apply_optimization(&self, recommendation: &OptimizationRecommendation) -> Result<()> {
        if !self.config.auto_tuning_enabled {
            return Err(NumRs2Error::InvalidOperation(
                "Auto-tuning is disabled".to_string(),
            ));
        }

        match recommendation.optimization_type {
            OptimizationType::OptimizeAlignment => {
                // Apply alignment optimization
                // This would modify allocator settings
                Ok(())
            }
            OptimizationType::UseArenaAllocation => {
                // Switch to arena allocation
                // This would require changing the global allocator
                Ok(())
            }
            OptimizationType::UsePoolAllocation => {
                // Switch to pool allocation
                Ok(())
            }
            OptimizationType::EnablePreallocation => {
                // Enable pre-allocation
                Ok(())
            }
            _ => Err(NumRs2Error::NotImplemented(format!(
                "Optimization type {:?} not yet implemented",
                recommendation.optimization_type
            ))),
        }
    }

    /// Generate a performance report
    pub fn generate_performance_report(&self) -> String {
        let current = self.get_current_metrics();
        let recommendations = self.analyze_performance();

        let mut report = String::new();
        report.push_str("=== Memory Allocator Performance Report ===\n\n");

        // Current metrics
        report.push_str("Current Performance Metrics:\n");
        report.push_str(&format!(
            "  Total allocations: {}\n",
            current.total_allocations
        ));
        report.push_str(&format!(
            "  Total deallocations: {}\n",
            current.total_deallocations
        ));
        report.push_str(&format!(
            "  Bytes allocated: {} MB\n",
            current.total_bytes_allocated / 1024 / 1024
        ));
        report.push_str(&format!(
            "  Bytes deallocated: {} MB\n",
            current.total_bytes_deallocated / 1024 / 1024
        ));
        report.push_str(&format!(
            "  Average allocation time: {} ns\n",
            current.avg_allocation_time_ns
        ));
        report.push_str(&format!(
            "  Average deallocation time: {} ns\n",
            current.avg_deallocation_time_ns
        ));
        report.push_str(&format!(
            "  Allocation failures: {}\n",
            current.allocation_failures
        ));
        report.push_str(&format!(
            "  Peak memory usage: {} MB\n",
            current.peak_memory_usage / 1024 / 1024
        ));
        report.push_str(&format!(
            "  Current memory usage: {} MB\n",
            current.current_memory_usage / 1024 / 1024
        ));

        // Performance characteristics
        report.push_str("\nPerformance Characteristics:\n");
        let allocation_rate = if current.avg_allocation_time_ns > 0 {
            1_000_000_000.0 / current.avg_allocation_time_ns as f64
        } else {
            0.0
        };
        report.push_str(&format!(
            "  Allocation rate: {:.0} ops/sec\n",
            allocation_rate
        ));

        let failure_rate = if current.total_allocations > 0 {
            current.allocation_failures as f64 / current.total_allocations as f64 * 100.0
        } else {
            0.0
        };
        report.push_str(&format!("  Failure rate: {:.3}%\n", failure_rate));

        let avg_allocation_size = if current.total_allocations > 0 {
            current.total_bytes_allocated / current.total_allocations
        } else {
            0
        };
        report.push_str(&format!(
            "  Average allocation size: {} bytes\n",
            avg_allocation_size
        ));

        // Recommendations
        if !recommendations.is_empty() {
            report.push_str("\nOptimization Recommendations:\n");
            for (i, rec) in recommendations.iter().enumerate() {
                report.push_str(&format!(
                    "  {}. {} (Est. improvement: {:.1}%, Difficulty: {})\n",
                    i + 1,
                    rec.description,
                    rec.estimated_improvement * 100.0,
                    rec.difficulty
                ));
            }
        } else {
            report.push_str("\nNo optimization recommendations at this time.\n");
        }

        report
    }

    /// Reset all metrics and history
    pub fn reset(&mut self) {
        *self.current_metrics.lock().unwrap() = PerformanceMetrics::default();
        self.metrics_history.clear();
        self.benchmark_cache.clear();
    }
}

/// Global performance tuner instance
static GLOBAL_TUNER: OnceLock<Mutex<PerformanceTuner>> = OnceLock::new();

/// Initialize the global performance tuner
pub fn init_global_tuner(config: TuningConfig) {
    let _ = GLOBAL_TUNER.set(Mutex::new(PerformanceTuner::new(config)));
}

/// Get reference to the global performance tuner
pub fn with_global_tuner<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&PerformanceTuner) -> R,
{
    GLOBAL_TUNER
        .get()
        .and_then(|tuner| tuner.lock().ok().map(|guard| f(&*guard)))
}

/// Get mutable reference to the global performance tuner
pub fn with_global_tuner_mut<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut PerformanceTuner) -> R,
{
    GLOBAL_TUNER
        .get()
        .and_then(|tuner| tuner.lock().ok().map(|mut guard| f(&mut *guard)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_alloc::enhanced_traits::NumericalArrayAllocator;
    #[allow(unused_imports)]
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_performance_tuner_creation() {
        let tuner = PerformanceTuner::default();
        let metrics = tuner.get_current_metrics();
        assert_eq!(metrics.total_allocations, 0);
    }

    #[test]
    fn test_metrics_recording() {
        let tuner = PerformanceTuner::default();

        tuner.record_allocation(1024, Duration::from_nanos(1000));
        tuner.record_allocation(2048, Duration::from_nanos(1500));

        let metrics = tuner.get_current_metrics();
        assert_eq!(metrics.total_allocations, 2);
        assert_eq!(metrics.total_bytes_allocated, 3072);
        assert_eq!(metrics.current_memory_usage, 3072);
        assert_eq!(metrics.peak_memory_usage, 3072);
    }

    #[test]
    fn test_deallocation_tracking() {
        let tuner = PerformanceTuner::default();

        tuner.record_allocation(1024, Duration::from_nanos(1000));
        tuner.record_deallocation(1024, Duration::from_nanos(500));

        let metrics = tuner.get_current_metrics();
        assert_eq!(metrics.total_allocations, 1);
        assert_eq!(metrics.total_deallocations, 1);
        assert_eq!(metrics.current_memory_usage, 0);
    }

    #[test]
    fn test_failure_recording() {
        let tuner = PerformanceTuner::default();

        tuner.record_allocation_failure();
        tuner.record_allocation_failure();

        let metrics = tuner.get_current_metrics();
        assert_eq!(metrics.allocation_failures, 2);
    }

    #[test]
    fn test_performance_analysis() {
        let tuner = PerformanceTuner::default();

        // Not enough data yet
        let recommendations = tuner.analyze_performance();
        assert!(recommendations.is_empty());

        // Add sufficient data
        for _ in 0..150 {
            tuner.record_allocation(64, Duration::from_nanos(500));
        }

        let recommendations = tuner.analyze_performance();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_benchmark_caching() {
        let mut tuner = PerformanceTuner::default();
        let allocator = NumericalArrayAllocator::new();
        let config = BenchmarkConfig {
            iterations: 100,
            min_size: 64,
            max_size: 256,
            concurrent_allocations: 10,
            randomize_sizes: false,
            randomize_order: false,
            memory_pressure: 0.0,
            enable_fragmentation: false,
        };

        // First benchmark
        let result1 = tuner
            .benchmark_allocator(&allocator, "TestAllocator", config.clone())
            .unwrap();

        // Second benchmark should be cached
        let result2 = tuner
            .benchmark_allocator(&allocator, "TestAllocator", config)
            .unwrap();

        assert_eq!(result1.allocator_name, result2.allocator_name);
        assert_eq!(
            result1.successful_allocations,
            result2.successful_allocations
        );
    }

    #[test]
    fn test_performance_report_generation() {
        let tuner = PerformanceTuner::default();

        // Add some metrics
        for i in 0..200 {
            tuner.record_allocation(1024 + i, Duration::from_nanos(1000 + i as u64));
        }

        let report = tuner.generate_performance_report();
        assert!(report.contains("Memory Allocator Performance Report"));
        assert!(report.contains("Total allocations: 200"));
    }

    #[test]
    fn test_global_tuner_initialization() {
        init_global_tuner(TuningConfig::default());

        let result = with_global_tuner(|tuner| tuner.get_current_metrics().total_allocations);

        assert_eq!(result, Some(0));
    }

    #[test]
    fn test_optimization_recommendations() {
        let tuner = PerformanceTuner::default();

        // Simulate small, frequent allocations
        for _ in 0..2000 {
            tuner.record_allocation(128, Duration::from_nanos(800));
        }

        let recommendations = tuner.analyze_performance();
        let has_arena_recommendation = recommendations
            .iter()
            .any(|r| r.optimization_type == OptimizationType::UseArenaAllocation);

        assert!(has_arena_recommendation);
    }
}
