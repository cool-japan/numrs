//! Performance-monitored allocator wrapper
//!
//! This module provides a wrapper that adds performance monitoring and automatic
//! tuning capabilities to any existing allocator.

use crate::error::{NumRs2Error, Result};
use crate::memory_alloc::performance_tuning::{PerformanceTuner, TuningConfig};
use crate::traits::{AllocationStrategy, MemoryAllocator, SpecializedAllocator};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// A performance-monitored allocator that wraps any existing allocator
pub struct MonitoredAllocator<A> {
    /// The underlying allocator
    inner: A,
    /// Performance tuner for monitoring and optimization
    tuner: Arc<Mutex<PerformanceTuner>>,
    /// Configuration for monitoring
    config: MonitoringConfig,
}

/// Configuration for allocation monitoring
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable detailed timing measurements
    pub detailed_timing: bool,
    /// Enable automatic optimization recommendations
    pub auto_optimization: bool,
    /// Minimum allocation size to monitor (smaller allocations are not tracked)
    pub min_monitored_size: usize,
    /// Maximum number of recent allocations to track for pattern analysis
    pub max_tracked_allocations: usize,
    /// How often to generate performance reports (in allocations)
    pub report_frequency: u64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            detailed_timing: true,
            auto_optimization: false,
            min_monitored_size: 64,
            max_tracked_allocations: 1000,
            report_frequency: 10000,
        }
    }
}

impl<A: std::fmt::Debug> std::fmt::Debug for MonitoredAllocator<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MonitoredAllocator")
            .field("inner", &self.inner)
            .field("config", &self.config)
            .field("tuner", &"Arc<Mutex<PerformanceTuner>>")
            .finish()
    }
}

impl<A> MonitoredAllocator<A>
where
    A: SpecializedAllocator<Error = NumRs2Error>,
{
    /// Create a new monitored allocator wrapping the given allocator
    pub fn new(inner: A, monitoring_config: MonitoringConfig, tuning_config: TuningConfig) -> Self {
        Self {
            inner,
            tuner: Arc::new(Mutex::new(PerformanceTuner::new(tuning_config))),
            config: monitoring_config,
        }
    }

    /// Create a monitored allocator with default configurations
    pub fn with_defaults(inner: A) -> Self {
        Self::new(inner, MonitoringConfig::default(), TuningConfig::default())
    }

    /// Get the current performance metrics
    pub fn get_performance_metrics(
        &self,
    ) -> crate::memory_alloc::performance_tuning::PerformanceMetrics {
        self.tuner
            .lock()
            .expect("tuner mutex should not be poisoned")
            .get_current_metrics()
    }

    /// Generate a performance report
    pub fn generate_performance_report(&self) -> String {
        self.tuner
            .lock()
            .expect("tuner mutex should not be poisoned")
            .generate_performance_report()
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(
        &self,
    ) -> Vec<crate::memory_alloc::performance_tuning::OptimizationRecommendation> {
        self.tuner
            .lock()
            .expect("tuner mutex should not be poisoned")
            .analyze_performance()
    }

    /// Reset performance metrics
    pub fn reset_metrics(&self) {
        self.tuner
            .lock()
            .expect("tuner mutex should not be poisoned")
            .reset();
    }

    /// Take a performance snapshot
    pub fn take_performance_snapshot(&self) {
        self.tuner
            .lock()
            .expect("tuner mutex should not be poisoned")
            .take_snapshot();
    }

    /// Enable or disable automatic optimization
    pub fn set_auto_optimization(&mut self, enabled: bool) {
        self.config.auto_optimization = enabled;
    }

    /// Check if allocation should be monitored based on size
    fn should_monitor_allocation(&self, layout: Layout) -> bool {
        layout.size() >= self.config.min_monitored_size
    }

    /// Record allocation metrics if monitoring is enabled
    fn record_allocation_metrics(
        &self,
        layout: Layout,
        duration: std::time::Duration,
        success: bool,
    ) {
        if !self.should_monitor_allocation(layout) {
            return;
        }

        let tuner = self
            .tuner
            .lock()
            .expect("tuner mutex should not be poisoned");
        if success {
            tuner.record_allocation(layout.size(), duration);
        } else {
            tuner.record_allocation_failure();
        }

        // Check if we should generate a report
        let metrics = tuner.get_current_metrics();
        if metrics
            .total_allocations
            .is_multiple_of(self.config.report_frequency)
        {
            println!("=== Allocation Performance Report ===");
            println!("{}", tuner.generate_performance_report());

            if self.config.auto_optimization {
                let recommendations = tuner.analyze_performance();
                if !recommendations.is_empty() {
                    println!("Auto-optimization recommendations:");
                    for rec in recommendations {
                        println!("  - {}", rec.description);
                    }
                }
            }
        }
    }

    /// Record deallocation metrics if monitoring is enabled
    fn record_deallocation_metrics(&self, layout: Layout, duration: std::time::Duration) {
        if !self.should_monitor_allocation(layout) {
            return;
        }

        let tuner = self
            .tuner
            .lock()
            .expect("tuner mutex should not be poisoned");
        tuner.record_deallocation(layout.size(), duration);
    }
}

impl<A> MemoryAllocator for MonitoredAllocator<A>
where
    A: SpecializedAllocator<Error = NumRs2Error>,
{
    type Error = NumRs2Error;

    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        let start_time = if self.config.detailed_timing && self.should_monitor_allocation(layout) {
            Some(Instant::now())
        } else {
            None
        };

        let result = self.inner.allocate(layout);

        if let Some(start) = start_time {
            let duration = start.elapsed();
            self.record_allocation_metrics(layout, duration, result.is_ok());
        }

        result
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        let start_time = if self.config.detailed_timing && self.should_monitor_allocation(layout) {
            Some(Instant::now())
        } else {
            None
        };

        let result = self.inner.deallocate(ptr, layout);

        if let Some(start) = start_time {
            let duration = start.elapsed();
            self.record_deallocation_metrics(layout, duration);
        }

        result
    }

    unsafe fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<u8>> {
        let start_time = if self.config.detailed_timing
            && (self.should_monitor_allocation(old_layout)
                || self.should_monitor_allocation(new_layout))
        {
            Some(Instant::now())
        } else {
            None
        };

        let result = self.inner.reallocate(ptr, old_layout, new_layout);

        if let Some(start) = start_time {
            let duration = start.elapsed();
            // Record as both deallocation and allocation
            self.record_deallocation_metrics(old_layout, duration);
            self.record_allocation_metrics(new_layout, duration, result.is_ok());
        }

        result
    }

    fn supports_layout(&self, layout: Layout) -> bool {
        self.inner.supports_layout(layout)
    }

    fn preferred_alignment(&self) -> usize {
        self.inner.preferred_alignment()
    }

    fn statistics(&self) -> Option<crate::traits::AllocationStats> {
        self.inner.statistics()
    }
}

impl<A> SpecializedAllocator for MonitoredAllocator<A>
where
    A: SpecializedAllocator<Error = NumRs2Error>,
{
    fn allocation_error(&self, msg: &str) -> Self::Error {
        self.inner.allocation_error(msg)
    }
}

impl<A> AllocationStrategy for MonitoredAllocator<A>
where
    A: SpecializedAllocator<Error = NumRs2Error> + AllocationStrategy,
{
    fn select_allocator(
        &self,
        _requirements: &crate::traits::AllocationRequirements,
    ) -> Box<dyn SpecializedAllocator<Error = NumRs2Error>> {
        // For monitored allocators, delegate to the underlying strategy would require
        // wrapping the result in monitoring, which is complex
        // For now, return the numerical array allocator which supports most use cases
        Box::new(crate::memory_alloc::enhanced_traits::NumericalArrayAllocator::new())
    }

    fn strategy_stats(&self) -> crate::traits::StrategyStats {
        crate::traits::StrategyStats::default()
    }
}

/// Factory for creating monitored allocators
pub struct MonitoredAllocatorFactory {
    monitoring_config: MonitoringConfig,
    tuning_config: TuningConfig,
}

impl Default for MonitoredAllocatorFactory {
    fn default() -> Self {
        Self::new(MonitoringConfig::default(), TuningConfig::default())
    }
}

impl MonitoredAllocatorFactory {
    /// Create a new factory with the given configurations
    pub fn new(monitoring_config: MonitoringConfig, tuning_config: TuningConfig) -> Self {
        Self {
            monitoring_config,
            tuning_config,
        }
    }

    /// Wrap any allocator with monitoring capabilities
    pub fn wrap<A>(&self, allocator: A) -> MonitoredAllocator<A>
    where
        A: SpecializedAllocator<Error = NumRs2Error>,
    {
        MonitoredAllocator::new(
            allocator,
            self.monitoring_config.clone(),
            self.tuning_config.clone(),
        )
    }

    /// Create a monitored numerical array allocator
    pub fn create_monitored_numerical(
        &self,
    ) -> MonitoredAllocator<crate::memory_alloc::enhanced_traits::NumericalArrayAllocator> {
        let numerical_allocator =
            crate::memory_alloc::enhanced_traits::NumericalArrayAllocator::new();
        self.wrap(numerical_allocator)
    }

    /// Create a monitored enhanced allocator bridge
    pub fn create_monitored_enhanced<T>(
        &self,
        inner: T,
    ) -> MonitoredAllocator<crate::memory_alloc::enhanced_traits::EnhancedAllocatorBridge<T>>
    where
        T: crate::memory_alloc::strategy::MemoryAllocator + std::fmt::Debug,
    {
        let enhanced_allocator =
            crate::memory_alloc::enhanced_traits::EnhancedAllocatorBridge::new(inner);
        self.wrap(enhanced_allocator)
    }

    /// Create a monitored allocator optimized for SIMD operations
    pub fn create_monitored_simd_optimized(
        &self,
    ) -> MonitoredAllocator<crate::memory_alloc::enhanced_traits::NumericalArrayAllocator> {
        // Use the numerical array allocator which is optimized for SIMD
        self.create_monitored_numerical()
    }
}

/// Convenience functions for creating commonly used monitored allocators
pub mod presets {
    use super::*;

    /// Create a high-performance monitored allocator for numerical computing
    pub fn numerical_computing_allocator(
    ) -> MonitoredAllocator<crate::memory_alloc::enhanced_traits::NumericalArrayAllocator> {
        let monitoring_config = MonitoringConfig {
            detailed_timing: true,
            auto_optimization: true,
            min_monitored_size: 256, // Monitor larger allocations typical in numerical computing
            max_tracked_allocations: 2000,
            report_frequency: 5000,
        };

        let tuning_config = TuningConfig {
            collection_interval_ms: 500,
            min_sample_size: 50,
            max_history_size: 500,
            improvement_threshold: 0.03, // 3% improvement threshold
            auto_tuning_enabled: true,
        };

        let factory = MonitoredAllocatorFactory::new(monitoring_config, tuning_config);
        factory.create_monitored_numerical()
    }

    /// Create a monitored allocator optimized for small temporary objects
    pub fn temporary_object_allocator(
    ) -> MonitoredAllocator<crate::memory_alloc::enhanced_traits::NumericalArrayAllocator> {
        let monitoring_config = MonitoringConfig {
            detailed_timing: false, // Less overhead for small allocations
            auto_optimization: true,
            min_monitored_size: 32,
            max_tracked_allocations: 5000,
            report_frequency: 20000,
        };

        let tuning_config = TuningConfig {
            collection_interval_ms: 2000,
            min_sample_size: 200,
            max_history_size: 1000,
            improvement_threshold: 0.05,
            auto_tuning_enabled: true,
        };

        let factory = MonitoredAllocatorFactory::new(monitoring_config, tuning_config);
        factory.create_monitored_numerical()
    }

    /// Create a monitored allocator for large matrix operations
    pub fn matrix_allocator(
        _typical_matrix_size: usize,
    ) -> MonitoredAllocator<crate::memory_alloc::enhanced_traits::NumericalArrayAllocator> {
        let monitoring_config = MonitoringConfig {
            detailed_timing: true,
            auto_optimization: false, // Matrix allocations have predictable patterns
            min_monitored_size: 1024, // Minimum 1KB for matrix monitoring
            max_tracked_allocations: 100,
            report_frequency: 1000,
        };

        let tuning_config = TuningConfig::default();
        let factory = MonitoredAllocatorFactory::new(monitoring_config, tuning_config);
        factory.create_monitored_numerical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_alloc::enhanced_traits::NumericalArrayAllocator;
    use std::alloc::Layout;

    #[test]
    fn test_monitored_allocator_creation() {
        let base_allocator = NumericalArrayAllocator::new();
        let monitored = MonitoredAllocator::with_defaults(base_allocator);

        let metrics = monitored.get_performance_metrics();
        assert_eq!(metrics.total_allocations, 0);
    }

    #[test]
    fn test_allocation_monitoring() {
        let base_allocator = NumericalArrayAllocator::new();
        let monitored = MonitoredAllocator::with_defaults(base_allocator);

        let layout = Layout::from_size_align(1024, 16).expect("Layout should succeed");
        let ptr = monitored
            .allocate(layout)
            .expect("allocation should succeed");

        let metrics = monitored.get_performance_metrics();
        assert_eq!(metrics.total_allocations, 1);
        assert_eq!(metrics.total_bytes_allocated, 1024);

        unsafe {
            monitored
                .deallocate(ptr, layout)
                .expect("deallocation should succeed");
        }

        let metrics = monitored.get_performance_metrics();
        assert_eq!(metrics.total_deallocations, 1);
        assert_eq!(metrics.current_memory_usage, 0);
    }

    #[test]
    fn test_small_allocation_filtering() {
        let config = MonitoringConfig {
            min_monitored_size: 1000,
            ..MonitoringConfig::default()
        };

        let base_allocator = NumericalArrayAllocator::new();
        let monitored = MonitoredAllocator::new(base_allocator, config, TuningConfig::default());

        // Small allocation should not be monitored
        let small_layout = Layout::from_size_align(64, 16).expect("Layout should succeed");
        let _ptr = monitored
            .allocate(small_layout)
            .expect("allocation should succeed");

        let metrics = monitored.get_performance_metrics();
        assert_eq!(metrics.total_allocations, 0);
    }

    #[test]
    fn test_performance_report_generation() {
        let base_allocator = NumericalArrayAllocator::new();
        let monitored = MonitoredAllocator::with_defaults(base_allocator);

        // Perform some allocations
        for _ in 0..10 {
            let layout = Layout::from_size_align(1024, 16).expect("Layout should succeed");
            let _ptr = monitored
                .allocate(layout)
                .expect("allocation should succeed");
        }

        let report = monitored.generate_performance_report();
        assert!(report.contains("Memory Allocator Performance Report"));
        assert!(report.contains("Total allocations: 10"));
    }

    #[test]
    fn test_factory_creation() {
        let factory = MonitoredAllocatorFactory::default();
        let base_allocator = NumericalArrayAllocator::new();
        let monitored = factory.wrap(base_allocator);

        assert!(monitored.preferred_alignment() >= 8);
    }

    #[test]
    fn test_preset_allocators() {
        let numerical = presets::numerical_computing_allocator();
        assert!(numerical.preferred_alignment() >= 8);

        let temp_obj = presets::temporary_object_allocator();
        assert!(temp_obj
            .supports_layout(Layout::from_size_align(64, 8).expect("Layout should succeed")));

        let matrix = presets::matrix_allocator(4096);
        assert!(matrix
            .supports_layout(Layout::from_size_align(4096, 8).expect("Layout should succeed")));
    }

    #[test]
    fn test_optimization_recommendations() {
        let base_allocator = NumericalArrayAllocator::new();
        let monitored = MonitoredAllocator::with_defaults(base_allocator);

        // Perform many small allocations to trigger recommendations
        for _ in 0..2000 {
            let layout = Layout::from_size_align(128, 8).expect("Layout should succeed");
            let ptr = monitored
                .allocate(layout)
                .expect("allocation should succeed");
            unsafe {
                monitored
                    .deallocate(ptr, layout)
                    .expect("deallocation should succeed");
            }
        }

        let recommendations = monitored.get_optimization_recommendations();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_metrics_reset() {
        let base_allocator = NumericalArrayAllocator::new();
        let monitored = MonitoredAllocator::with_defaults(base_allocator);

        // Perform allocation
        let layout = Layout::from_size_align(1024, 8).expect("Layout should succeed");
        let _ptr = monitored
            .allocate(layout)
            .expect("allocation should succeed");

        let metrics_before = monitored.get_performance_metrics();
        assert_eq!(metrics_before.total_allocations, 1);

        // Reset metrics
        monitored.reset_metrics();

        let metrics_after = monitored.get_performance_metrics();
        assert_eq!(metrics_after.total_allocations, 0);
    }
}
