//! Memory allocator benchmarking and performance analysis
//!
//! This module provides comprehensive benchmarking tools for evaluating and comparing
//! memory allocator performance across different workload patterns.

use crate::error::{NumRs2Error, Result};
use crate::traits::SpecializedAllocator;
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Benchmarking configuration for memory allocator performance testing
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of allocation/deallocation cycles to perform
    pub iterations: usize,
    /// Minimum allocation size in bytes
    pub min_size: usize,
    /// Maximum allocation size in bytes
    pub max_size: usize,
    /// Number of concurrent allocations to maintain
    pub concurrent_allocations: usize,
    /// Whether to randomize allocation sizes
    pub randomize_sizes: bool,
    /// Whether to randomize allocation order
    pub randomize_order: bool,
    /// Memory pressure simulation (0.0 = no pressure, 1.0 = maximum pressure)
    pub memory_pressure: f64,
    /// Whether to fragment memory intentionally
    pub enable_fragmentation: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 10000,
            min_size: 64,
            max_size: 4096,
            concurrent_allocations: 100,
            randomize_sizes: true,
            randomize_order: true,
            memory_pressure: 0.0,
            enable_fragmentation: false,
        }
    }
}

/// Performance metrics collected during allocator benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Allocator name being benchmarked
    pub allocator_name: String,
    /// Configuration used for benchmarking
    pub config: BenchmarkConfig,
    /// Total time spent on allocations
    pub allocation_time: Duration,
    /// Total time spent on deallocations
    pub deallocation_time: Duration,
    /// Average allocation time per operation
    pub avg_allocation_time: Duration,
    /// Average deallocation time per operation
    pub avg_deallocation_time: Duration,
    /// Peak memory usage during benchmark
    pub peak_memory_usage: usize,
    /// Total bytes allocated during benchmark
    pub total_bytes_allocated: usize,
    /// Number of successful allocations
    pub successful_allocations: usize,
    /// Number of failed allocations
    pub failed_allocations: usize,
    /// Memory fragmentation level (0.0-1.0)
    pub fragmentation_level: f64,
    /// Allocation efficiency (useful bytes / total bytes allocated)
    pub allocation_efficiency: f64,
    /// Throughput in allocations per second
    pub allocation_throughput: f64,
    /// Throughput in bytes per second
    pub bytes_per_second: f64,
    /// Distribution of allocation sizes
    pub size_distribution: HashMap<usize, usize>,
    /// Latency percentiles for allocation times (50th, 90th, 95th, 99th)
    pub latency_percentiles: (Duration, Duration, Duration, Duration),
}

impl fmt::Display for BenchmarkResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "=== Allocator Benchmark Results: {} ===",
            self.allocator_name
        )?;
        writeln!(f, "Configuration:")?;
        writeln!(f, "  Iterations: {}", self.config.iterations)?;
        writeln!(
            f,
            "  Size range: {} - {} bytes",
            self.config.min_size, self.config.max_size
        )?;
        writeln!(
            f,
            "  Concurrent allocations: {}",
            self.config.concurrent_allocations
        )?;
        writeln!(
            f,
            "  Memory pressure: {:.1}%",
            self.config.memory_pressure * 100.0
        )?;
        writeln!(f)?;
        writeln!(f, "Performance Metrics:")?;
        writeln!(f, "  Total allocation time: {:?}", self.allocation_time)?;
        writeln!(f, "  Total deallocation time: {:?}", self.deallocation_time)?;
        writeln!(
            f,
            "  Average allocation time: {:?}",
            self.avg_allocation_time
        )?;
        writeln!(
            f,
            "  Average deallocation time: {:?}",
            self.avg_deallocation_time
        )?;
        writeln!(
            f,
            "  Peak memory usage: {} MB",
            self.peak_memory_usage / 1024 / 1024
        )?;
        writeln!(
            f,
            "  Total bytes allocated: {} MB",
            self.total_bytes_allocated / 1024 / 1024
        )?;
        writeln!(
            f,
            "  Successful allocations: {}",
            self.successful_allocations
        )?;
        writeln!(f, "  Failed allocations: {}", self.failed_allocations)?;
        writeln!(f, "  Fragmentation level: {:.3}", self.fragmentation_level)?;
        writeln!(
            f,
            "  Allocation efficiency: {:.3}",
            self.allocation_efficiency
        )?;
        writeln!(
            f,
            "  Allocation throughput: {:.0} ops/sec",
            self.allocation_throughput
        )?;
        writeln!(
            f,
            "  Bytes throughput: {:.2} MB/sec",
            self.bytes_per_second / 1024.0 / 1024.0
        )?;
        writeln!(f)?;
        writeln!(f, "Latency Percentiles:")?;
        writeln!(f, "  50th: {:?}", self.latency_percentiles.0)?;
        writeln!(f, "  90th: {:?}", self.latency_percentiles.1)?;
        writeln!(f, "  95th: {:?}", self.latency_percentiles.2)?;
        writeln!(f, "  99th: {:?}", self.latency_percentiles.3)?;
        Ok(())
    }
}

/// Memory allocator benchmarking framework
pub struct AllocatorBenchmark {
    config: BenchmarkConfig,
    rng_state: u64, // Simple XORShift PRNG for reproducible benchmarks
}

impl AllocatorBenchmark {
    /// Create a new allocator benchmark with the given configuration
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            rng_state: 0x123456789abcdef0,
        }
    }

    /// Create a benchmark with default configuration
    pub fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Simple XORShift PRNG for reproducible random numbers
    fn next_random(&mut self) -> u64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        self.rng_state
    }

    /// Generate a random allocation size within the configured range
    fn random_size(&mut self) -> usize {
        if !self.config.randomize_sizes {
            return (self.config.min_size + self.config.max_size) / 2;
        }

        let range = self.config.max_size - self.config.min_size;
        if range == 0 {
            return self.config.min_size;
        }

        let random_offset = (self.next_random() as usize) % range;
        self.config.min_size + random_offset
    }

    /// Benchmark a specific allocator implementation
    pub fn benchmark_allocator<A>(&mut self, allocator: &A, name: &str) -> Result<BenchmarkResults>
    where
        A: SpecializedAllocator<Error = NumRs2Error> + ?Sized,
    {
        let mut results = BenchmarkResults {
            allocator_name: name.to_string(),
            config: self.config.clone(),
            allocation_time: Duration::ZERO,
            deallocation_time: Duration::ZERO,
            avg_allocation_time: Duration::ZERO,
            avg_deallocation_time: Duration::ZERO,
            peak_memory_usage: 0,
            total_bytes_allocated: 0,
            successful_allocations: 0,
            failed_allocations: 0,
            fragmentation_level: 0.0,
            allocation_efficiency: 0.0,
            allocation_throughput: 0.0,
            bytes_per_second: 0.0,
            size_distribution: HashMap::new(),
            latency_percentiles: (
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
            ),
        };

        let mut active_allocations: Vec<(std::ptr::NonNull<u8>, std::alloc::Layout)> = Vec::new();
        let mut allocation_times: Vec<Duration> = Vec::new();
        let mut deallocation_times: Vec<Duration> = Vec::new();
        let mut current_memory_usage = 0usize;

        // Warm-up phase
        self.warmup_allocator(allocator)?;

        let benchmark_start = Instant::now();

        // Main benchmark loop
        for iteration in 0..self.config.iterations {
            // Determine if we should allocate or deallocate
            let should_allocate = active_allocations.len() < self.config.concurrent_allocations
                || (active_allocations.len() < self.config.concurrent_allocations * 2
                    && self.next_random() % 2 == 0);

            if should_allocate {
                // Perform allocation
                let size = self.random_size();
                let align = if size >= 32 {
                    32
                } else {
                    std::mem::align_of::<usize>()
                };

                match std::alloc::Layout::from_size_align(size, align) {
                    Ok(layout) => {
                        let alloc_start = Instant::now();
                        match allocator.allocate(layout) {
                            Ok(ptr) => {
                                let alloc_time = alloc_start.elapsed();
                                allocation_times.push(alloc_time);
                                results.allocation_time += alloc_time;
                                results.successful_allocations += 1;
                                results.total_bytes_allocated += size;
                                current_memory_usage += size;
                                results.peak_memory_usage =
                                    results.peak_memory_usage.max(current_memory_usage);

                                *results.size_distribution.entry(size).or_insert(0) += 1;
                                active_allocations.push((ptr, layout));
                            }
                            Err(_) => {
                                results.failed_allocations += 1;
                            }
                        }
                    }
                    Err(_) => {
                        results.failed_allocations += 1;
                    }
                }
            } else if !active_allocations.is_empty() {
                // Perform deallocation
                let index = if self.config.randomize_order {
                    (self.next_random() as usize) % active_allocations.len()
                } else {
                    0 // FIFO order
                };

                let (ptr, layout) = active_allocations.remove(index);
                let dealloc_start = Instant::now();

                unsafe {
                    let _ = allocator.deallocate(ptr, layout);
                }

                let dealloc_time = dealloc_start.elapsed();
                deallocation_times.push(dealloc_time);
                results.deallocation_time += dealloc_time;
                current_memory_usage -= layout.size();
            }

            // Apply memory pressure simulation
            if self.config.memory_pressure > 0.0 && iteration % 100 == 0 {
                self.apply_memory_pressure(&mut active_allocations, allocator)?;
            }
        }

        // Clean up remaining allocations
        for (ptr, layout) in active_allocations {
            let dealloc_start = Instant::now();
            unsafe {
                let _ = allocator.deallocate(ptr, layout);
            }
            let dealloc_time = dealloc_start.elapsed();
            deallocation_times.push(dealloc_time);
            results.deallocation_time += dealloc_time;
        }

        let total_benchmark_time = benchmark_start.elapsed();

        // Calculate derived metrics
        self.calculate_derived_metrics(
            &mut results,
            &allocation_times,
            &deallocation_times,
            total_benchmark_time,
        );

        Ok(results)
    }

    /// Warm up the allocator to avoid cold-start effects in benchmarks
    fn warmup_allocator<A>(&mut self, allocator: &A) -> Result<()>
    where
        A: SpecializedAllocator<Error = NumRs2Error> + ?Sized,
    {
        let warmup_iterations = std::cmp::min(1000, self.config.iterations / 10);
        let mut warmup_allocations = Vec::new();

        // Perform some warm-up allocations
        for _ in 0..warmup_iterations {
            let size = self.random_size();
            let align = std::mem::align_of::<usize>();

            if let Ok(layout) = std::alloc::Layout::from_size_align(size, align) {
                if let Ok(ptr) = allocator.allocate(layout) {
                    warmup_allocations.push((ptr, layout));
                }
            }
        }

        // Clean up warm-up allocations
        for (ptr, layout) in warmup_allocations {
            unsafe {
                let _ = allocator.deallocate(ptr, layout);
            }
        }

        Ok(())
    }

    /// Apply memory pressure simulation
    fn apply_memory_pressure<A>(
        &mut self,
        active_allocations: &mut Vec<(std::ptr::NonNull<u8>, std::alloc::Layout)>,
        allocator: &A,
    ) -> Result<()>
    where
        A: SpecializedAllocator<Error = NumRs2Error> + ?Sized,
    {
        if self.config.memory_pressure <= 0.0 {
            return Ok(());
        }

        // Randomly deallocate some allocations to simulate pressure
        let deallocations_to_perform =
            ((active_allocations.len() as f64 * self.config.memory_pressure) as usize).max(1);

        for _ in 0..deallocations_to_perform.min(active_allocations.len()) {
            if !active_allocations.is_empty() {
                let index = (self.next_random() as usize) % active_allocations.len();
                let (ptr, layout) = active_allocations.remove(index);
                unsafe {
                    let _ = allocator.deallocate(ptr, layout);
                }
            }
        }

        Ok(())
    }

    /// Calculate derived performance metrics
    fn calculate_derived_metrics(
        &self,
        results: &mut BenchmarkResults,
        allocation_times: &[Duration],
        deallocation_times: &[Duration],
        total_time: Duration,
    ) {
        // Calculate average times
        if results.successful_allocations > 0 {
            results.avg_allocation_time =
                results.allocation_time / results.successful_allocations as u32;
        }

        if !deallocation_times.is_empty() {
            results.avg_deallocation_time =
                results.deallocation_time / deallocation_times.len() as u32;
        }

        // Calculate throughput
        let total_seconds = total_time.as_secs_f64();
        if total_seconds > 0.0 {
            results.allocation_throughput = results.successful_allocations as f64 / total_seconds;
            results.bytes_per_second = results.total_bytes_allocated as f64 / total_seconds;
        }

        // Calculate allocation efficiency (assuming minimal overhead)
        let total_allocations = results.successful_allocations + results.failed_allocations;
        if total_allocations > 0 {
            results.allocation_efficiency =
                results.successful_allocations as f64 / total_allocations as f64;
        }

        // Calculate latency percentiles
        if !allocation_times.is_empty() {
            let mut sorted_times = allocation_times.to_vec();
            sorted_times.sort();

            let len = sorted_times.len();
            results.latency_percentiles = (
                sorted_times[len / 2],          // 50th percentile
                sorted_times[(len * 9) / 10],   // 90th percentile
                sorted_times[(len * 95) / 100], // 95th percentile
                sorted_times[(len * 99) / 100], // 99th percentile
            );
        }

        // Estimate fragmentation (simplified heuristic)
        if results.peak_memory_usage > 0 && results.total_bytes_allocated > 0 {
            // Higher values indicate more fragmentation
            results.fragmentation_level =
                1.0 - (results.total_bytes_allocated as f64 / results.peak_memory_usage as f64);
            results.fragmentation_level = results.fragmentation_level.max(0.0).min(1.0);
        }
    }

    /// Compare multiple allocators with the same benchmark configuration
    pub fn compare_allocators(
        &mut self,
        allocators: Vec<(Box<dyn SpecializedAllocator<Error = NumRs2Error>>, String)>,
    ) -> Result<Vec<BenchmarkResults>> {
        let mut all_results = Vec::new();

        for (allocator, name) in allocators {
            let results = self.benchmark_allocator(allocator.as_ref(), &name)?;
            all_results.push(results);
        }

        Ok(all_results)
    }

    /// Generate a performance report comparing multiple benchmark results
    pub fn generate_comparison_report(results: &[BenchmarkResults]) -> String {
        let mut report = String::new();

        report.push_str("=== Allocator Performance Comparison ===\n\n");

        if results.is_empty() {
            report.push_str("No benchmark results to compare.\n");
            return report;
        }

        // Summary table
        report.push_str("Performance Summary:\n");
        report.push_str(&format!(
            "{:<20} {:<15} {:<15} {:<15} {:<15} {:<10}\n",
            "Allocator", "Alloc Time", "Dealloc Time", "Throughput", "Efficiency", "Frag"
        ));
        report.push_str(&"-".repeat(100));
        report.push('\n');

        for result in results {
            report.push_str(&format!(
                "{:<20} {:<15.2?} {:<15.2?} {:<15.0} {:<15.3} {:<10.3}\n",
                result.allocator_name,
                result.avg_allocation_time,
                result.avg_deallocation_time,
                result.allocation_throughput,
                result.allocation_efficiency,
                result.fragmentation_level
            ));
        }

        // Find best performer in each category
        report.push_str("\n\nBest Performers:\n");

        if let Some(fastest_alloc) = results.iter().min_by_key(|r| r.avg_allocation_time) {
            report.push_str(&format!(
                "Fastest Allocation: {} ({:?})\n",
                fastest_alloc.allocator_name, fastest_alloc.avg_allocation_time
            ));
        }

        if let Some(fastest_dealloc) = results.iter().min_by_key(|r| r.avg_deallocation_time) {
            report.push_str(&format!(
                "Fastest Deallocation: {} ({:?})\n",
                fastest_dealloc.allocator_name, fastest_dealloc.avg_deallocation_time
            ));
        }

        if let Some(highest_throughput) = results.iter().max_by(|a, b| {
            a.allocation_throughput
                .partial_cmp(&b.allocation_throughput)
                .unwrap()
        }) {
            report.push_str(&format!(
                "Highest Throughput: {} ({:.0} ops/sec)\n",
                highest_throughput.allocator_name, highest_throughput.allocation_throughput
            ));
        }

        if let Some(most_efficient) = results.iter().max_by(|a, b| {
            a.allocation_efficiency
                .partial_cmp(&b.allocation_efficiency)
                .unwrap()
        }) {
            report.push_str(&format!(
                "Most Efficient: {} ({:.3})\n",
                most_efficient.allocator_name, most_efficient.allocation_efficiency
            ));
        }

        if let Some(least_fragmented) = results.iter().min_by(|a, b| {
            a.fragmentation_level
                .partial_cmp(&b.fragmentation_level)
                .unwrap()
        }) {
            report.push_str(&format!(
                "Least Fragmentation: {} ({:.3})\n",
                least_fragmented.allocator_name, least_fragmented.fragmentation_level
            ));
        }

        report
    }
}

/// Predefined benchmark configurations for common scenarios
pub mod benchmark_configs {
    use super::BenchmarkConfig;

    /// Small frequent allocations (typical for temporary calculations)
    pub fn small_frequent() -> BenchmarkConfig {
        BenchmarkConfig {
            iterations: 50000,
            min_size: 16,
            max_size: 256,
            concurrent_allocations: 50,
            randomize_sizes: true,
            randomize_order: true,
            memory_pressure: 0.1,
            enable_fragmentation: false,
        }
    }

    /// Large matrix allocations (typical for numerical computing)
    pub fn large_matrices() -> BenchmarkConfig {
        BenchmarkConfig {
            iterations: 1000,
            min_size: 1024 * 1024,      // 1MB
            max_size: 64 * 1024 * 1024, // 64MB
            concurrent_allocations: 10,
            randomize_sizes: true,
            randomize_order: false,
            memory_pressure: 0.0,
            enable_fragmentation: false,
        }
    }

    /// Mixed workload (combination of small and large allocations)
    pub fn mixed_workload() -> BenchmarkConfig {
        BenchmarkConfig {
            iterations: 10000,
            min_size: 64,
            max_size: 4 * 1024 * 1024, // 4MB
            concurrent_allocations: 100,
            randomize_sizes: true,
            randomize_order: true,
            memory_pressure: 0.2,
            enable_fragmentation: true,
        }
    }

    /// High memory pressure scenario
    pub fn high_pressure() -> BenchmarkConfig {
        BenchmarkConfig {
            iterations: 20000,
            min_size: 1024,
            max_size: 16 * 1024,
            concurrent_allocations: 200,
            randomize_sizes: true,
            randomize_order: true,
            memory_pressure: 0.8,
            enable_fragmentation: true,
        }
    }

    /// SIMD-aligned allocations
    pub fn simd_aligned() -> BenchmarkConfig {
        BenchmarkConfig {
            iterations: 5000,
            min_size: 256,
            max_size: 8192,
            concurrent_allocations: 50,
            randomize_sizes: false, // Use consistent sizes for SIMD
            randomize_order: false,
            memory_pressure: 0.0,
            enable_fragmentation: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_alloc::enhanced_traits::NumericalArrayAllocator;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.iterations, 10000);
        assert_eq!(config.min_size, 64);
        assert_eq!(config.max_size, 4096);
        assert!(config.randomize_sizes);
    }

    #[test]
    fn test_allocator_benchmark_creation() {
        let config = BenchmarkConfig::default();
        let benchmark = AllocatorBenchmark::new(config);
        assert_eq!(benchmark.config.iterations, 10000);
    }

    #[test]
    fn test_benchmark_aligned_allocator() {
        let mut benchmark = AllocatorBenchmark::new(BenchmarkConfig {
            iterations: 100,
            min_size: 64,
            max_size: 256,
            concurrent_allocations: 10,
            randomize_sizes: false,
            randomize_order: false,
            memory_pressure: 0.0,
            enable_fragmentation: false,
        });

        let allocator = NumericalArrayAllocator::new();
        let results = benchmark
            .benchmark_allocator(&allocator, "NumericalArrayAllocator")
            .unwrap();

        assert_eq!(results.allocator_name, "NumericalArrayAllocator");
        assert!(results.successful_allocations > 0);
        assert!(results.allocation_throughput > 0.0);
    }

    #[test]
    fn test_predefined_configs() {
        let small_config = benchmark_configs::small_frequent();
        assert_eq!(small_config.min_size, 16);
        assert_eq!(small_config.max_size, 256);

        let large_config = benchmark_configs::large_matrices();
        assert_eq!(large_config.min_size, 1024 * 1024);
        assert_eq!(large_config.max_size, 64 * 1024 * 1024);
    }

    #[test]
    fn test_comparison_report_generation() {
        let results = vec![BenchmarkResults {
            allocator_name: "TestAllocator1".to_string(),
            config: BenchmarkConfig::default(),
            allocation_time: Duration::from_millis(100),
            deallocation_time: Duration::from_millis(50),
            avg_allocation_time: Duration::from_micros(10),
            avg_deallocation_time: Duration::from_micros(5),
            peak_memory_usage: 1024 * 1024,
            total_bytes_allocated: 800 * 1024,
            successful_allocations: 10000,
            failed_allocations: 0,
            fragmentation_level: 0.2,
            allocation_efficiency: 1.0,
            allocation_throughput: 100000.0,
            bytes_per_second: 8000000.0,
            size_distribution: HashMap::new(),
            latency_percentiles: (
                Duration::from_micros(8),
                Duration::from_micros(15),
                Duration::from_micros(20),
                Duration::from_micros(30),
            ),
        }];

        let report = AllocatorBenchmark::generate_comparison_report(&results);
        assert!(report.contains("Allocator Performance Comparison"));
        assert!(report.contains("TestAllocator1"));
        assert!(report.contains("Best Performers"));
    }
}
