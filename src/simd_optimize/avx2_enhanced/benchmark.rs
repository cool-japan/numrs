//! Benchmarking utilities for SIMD operations
//!
//! This module provides:
//! - SimdPerformanceMonitor: Track SIMD operation performance
//! - SimdBenchmark: Compare scalar vs SIMD implementations
//! - SimdBenchmarkResults: Store and display benchmark results

use crate::array::Array;
use crate::simd::SimdOps;

/// Performance monitoring for SIMD operations
#[derive(Debug, Default, Clone)]
pub struct SimdPerformanceMonitor {
    /// Total number of operations performed
    pub operations_count: usize,
    /// Total elements processed
    pub total_elements: usize,
    /// Total elements processed with SIMD
    pub vectorized_elements: usize,
    /// Ratio of vectorized to total operations
    pub vectorization_ratio: f64,
}

impl SimdPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a SIMD operation
    pub fn record_operation(&mut self, total_elements: usize, vectorized_elements: usize) {
        self.operations_count += 1;
        self.total_elements += total_elements;
        self.vectorized_elements += vectorized_elements;
        self.vectorization_ratio = self.vectorized_elements as f64 / self.total_elements as f64;
    }

    /// Get statistics summary
    pub fn get_summary(&self) -> String {
        format!(
            "Operations: {}, Total Elements: {}, Vectorized: {} ({:.1}%)",
            self.operations_count,
            self.total_elements,
            self.vectorized_elements,
            self.vectorization_ratio * 100.0
        )
    }

    /// Reset all counters
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Benchmark utilities for comparing implementations
pub struct SimdBenchmark;

impl SimdBenchmark {
    /// Benchmark different SIMD implementations
    pub fn compare_implementations(size: usize, iterations: usize) -> SimdBenchmarkResults {
        use std::time::Instant;

        let data1 = Array::from_vec((0..size).map(|i| i as f32).collect::<Vec<_>>());
        let data2 = Array::from_vec((0..size).map(|i| (i + 1) as f32).collect::<Vec<_>>());

        // Benchmark scalar addition
        let start = Instant::now();
        for _ in 0..iterations {
            let _result = data1.add(&data2);
        }
        let scalar_time = start.elapsed().as_nanos() as f64;

        // Benchmark SIMD addition
        let start = Instant::now();
        for _ in 0..iterations {
            let _result = data1.simd_add(&data2).expect("SIMD add should succeed");
        }
        let simd_time = start.elapsed().as_nanos() as f64;

        SimdBenchmarkResults {
            scalar_time_ns: scalar_time / iterations as f64,
            simd_time_ns: simd_time / iterations as f64,
            speedup: scalar_time / simd_time,
            elements: size,
            throughput_elements_per_ns: size as f64 / (simd_time / iterations as f64),
        }
    }

    /// Benchmark a specific operation
    pub fn benchmark_operation<F, T>(f: F, iterations: usize) -> BenchmarkResult
    where
        F: Fn() -> T,
    {
        use std::time::Instant;

        // Warmup
        for _ in 0..5 {
            let _ = f();
        }

        // Actual benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = f();
        }
        let elapsed = start.elapsed();

        BenchmarkResult {
            total_time_ns: elapsed.as_nanos() as f64,
            avg_time_ns: elapsed.as_nanos() as f64 / iterations as f64,
            iterations,
        }
    }
}

/// Results from SIMD benchmark comparison
#[derive(Debug, Clone)]
pub struct SimdBenchmarkResults {
    /// Average time for scalar implementation in nanoseconds
    pub scalar_time_ns: f64,
    /// Average time for SIMD implementation in nanoseconds
    pub simd_time_ns: f64,
    /// Speedup ratio (scalar_time / simd_time)
    pub speedup: f64,
    /// Number of elements processed
    pub elements: usize,
    /// Throughput in elements per nanosecond
    pub throughput_elements_per_ns: f64,
}

impl SimdBenchmarkResults {
    /// Print a summary of the benchmark results
    pub fn print_summary(&self) {
        println!("SIMD Benchmark Results:");
        println!("  Elements: {}", self.elements);
        println!("  Scalar time: {:.2} ns", self.scalar_time_ns);
        println!("  SIMD time: {:.2} ns", self.simd_time_ns);
        println!("  Speedup: {:.2}x", self.speedup);
        println!(
            "  Throughput: {:.2} elements/ns",
            self.throughput_elements_per_ns
        );
    }

    /// Get speedup as a formatted string
    pub fn speedup_str(&self) -> String {
        format!("{:.2}x", self.speedup)
    }
}

/// Result from a single benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Total time in nanoseconds
    pub total_time_ns: f64,
    /// Average time per iteration in nanoseconds
    pub avg_time_ns: f64,
    /// Number of iterations
    pub iterations: usize,
}

impl BenchmarkResult {
    /// Get operations per second
    pub fn ops_per_second(&self) -> f64 {
        1_000_000_000.0 / self.avg_time_ns
    }

    /// Print summary
    pub fn print_summary(&self, name: &str) {
        println!("Benchmark: {}", name);
        println!("  Iterations: {}", self.iterations);
        println!("  Average time: {:.2} ns", self.avg_time_ns);
        println!("  Ops/second: {:.2}", self.ops_per_second());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_basic() {
        let mut monitor = SimdPerformanceMonitor::new();

        monitor.record_operation(1000, 800);
        assert_eq!(monitor.operations_count, 1);
        assert_eq!(monitor.total_elements, 1000);
        assert_eq!(monitor.vectorized_elements, 800);

        monitor.record_operation(500, 400);
        assert_eq!(monitor.operations_count, 2);
        assert_eq!(monitor.total_elements, 1500);
        assert_eq!(monitor.vectorized_elements, 1200);
    }

    #[test]
    fn test_performance_monitor_reset() {
        let mut monitor = SimdPerformanceMonitor::new();
        monitor.record_operation(1000, 800);
        monitor.reset();

        assert_eq!(monitor.operations_count, 0);
        assert_eq!(monitor.total_elements, 0);
    }

    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult {
            total_time_ns: 1_000_000.0,
            avg_time_ns: 1000.0,
            iterations: 1000,
        };

        let ops = result.ops_per_second();
        assert!(ops > 0.0);
    }
}
