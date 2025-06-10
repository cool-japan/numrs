//! Parallelization threshold optimization
//!
//! This module provides functionality for determining optimal
//! thresholds for when to use parallel computation.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// Global threshold value (atomic for thread safety)
static GLOBAL_THRESHOLD: AtomicUsize = AtomicUsize::new(1000);

/// Types of parallelization thresholds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelizationThreshold {
    /// Fixed threshold value
    Fixed(usize),
    /// Adaptive threshold based on CPU and memory characteristics
    Adaptive,
    /// Threshold determined by benchmarking on the current hardware
    Benchmarked,
    /// Experimental auto-tuning threshold that adjusts based on runtime performance
    AutoTuning,
}

/// Set the global threshold for parallelization
///
/// # Arguments
///
/// * `threshold` - The new threshold value
pub fn set_global_threshold(threshold: usize) {
    GLOBAL_THRESHOLD.store(threshold, Ordering::SeqCst);
}

/// Get the current global threshold for parallelization
pub fn get_global_threshold() -> usize {
    GLOBAL_THRESHOLD.load(Ordering::SeqCst)
}

/// Calculate an adaptive threshold based on array size and element cost
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `element_cost` - The computational cost per element (in arbitrary units)
///
/// # Returns
///
/// The adaptive threshold value
pub fn adaptive_threshold(array_size: usize, element_cost: f64) -> usize {
    let core_count = num_cpus::get();

    // Base threshold depends on number of cores
    let base_threshold = match core_count {
        1 => usize::MAX, // Never parallelize on a single core
        2 => 5000,       // Dual core - higher threshold
        3..=4 => 2000,   // Quad core - medium threshold
        5..=8 => 1000,   // Octa core - lower threshold
        _ => 500,        // Many cores - very low threshold
    };

    // Adjust for element cost - higher cost means lower threshold
    let cost_factor = (5.0 / element_cost).clamp(0.1, 10.0);

    // Adjust for array size - very large arrays may benefit from different thresholds
    let size_factor = if array_size > 1_000_000 {
        0.8 // Lower threshold for very large arrays
    } else if array_size > 100_000 {
        0.9 // Slightly lower threshold for large arrays
    } else {
        1.0 // Standard threshold for normal-sized arrays
    };

    // Calculate the final threshold
    let calculated = (base_threshold as f64 * cost_factor * size_factor) as usize;

    // Ensure the threshold is reasonable
    calculated.clamp(100, 50_000)
}

/// Benchmark to determine the optimal threshold for the current hardware
///
/// # Arguments
///
/// * `element_cost` - A function that simulates the cost of processing a single element
///
/// # Returns
///
/// The optimal threshold determined by benchmarking
pub fn benchmark_threshold<F>(element_cost: F) -> usize
where
    F: Fn(usize) -> f64 + Sync + Send,
{
    // Function to benchmark parallel vs sequential performance
    let benchmark = |size: usize| -> (Duration, Duration) {
        let data: Vec<usize> = (0..size).collect();

        // Benchmark sequential
        let start = Instant::now();
        let _sequential: f64 = data.iter().map(|&i| element_cost(i)).sum();
        let sequential_time = start.elapsed();

        // Benchmark parallel
        let start = Instant::now();
        let _parallel: f64 = data.par_iter().map(|&i| element_cost(i)).sum();
        let parallel_time = start.elapsed();

        (sequential_time, parallel_time)
    };

    // Test different sizes to find the threshold
    let sizes = [100, 500, 1000, 2000, 5000, 10000, 20000];
    let mut threshold = 1000; // Default

    for &size in &sizes {
        let (seq_time, par_time) = benchmark(size);

        // If parallel is faster, return the previous size as threshold
        if par_time < seq_time {
            return threshold;
        }

        threshold = size;
    }

    // If no threshold found, use the largest tested size
    threshold
}

/// Get the optimal threshold based on the specified strategy
///
/// # Arguments
///
/// * `threshold_type` - The type of threshold to use
/// * `array_size` - The size of the array to process
/// * `element_cost` - The computational cost per element (in arbitrary units)
///
/// # Returns
///
/// The optimal threshold value
pub fn get_optimal_threshold(
    threshold_type: ParallelizationThreshold,
    array_size: usize,
    element_cost: f64,
) -> usize {
    match threshold_type {
        ParallelizationThreshold::Fixed(value) => value,
        ParallelizationThreshold::Adaptive => adaptive_threshold(array_size, element_cost),
        ParallelizationThreshold::Benchmarked => {
            // Use a simple benchmark to determine threshold
            // In a real implementation, this would use more sophisticated benchmarking
            benchmark_threshold(|i| {
                // Simulate the element cost with a simple calculation
                let mut result = i as f64;
                for _ in 0..(element_cost as usize).max(1) {
                    result = result.sin().cos();
                }
                result
            })
        }
        ParallelizationThreshold::AutoTuning => {
            // In a real implementation, this would track performance and adjust the threshold
            // For now, just use the global threshold which could be set by auto-tuning code
            get_global_threshold()
        }
    }
}

/// Auto-tune the parallelization threshold based on runtime performance
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `element_cost` - The computational cost per element (in arbitrary units)
///
/// # Returns
///
/// The auto-tuned threshold value
pub fn auto_tune_threshold(array_size: usize, element_cost: f64) -> usize {
    let current = get_global_threshold();

    // Perform simple auto-tuning
    // In a real implementation, this would be more sophisticated
    let new_threshold = if array_size > current && element_cost > 1.0 {
        // If the array is larger than current threshold and element cost is high,
        // decrease the threshold slightly to favor parallelization
        (current as f64 * 0.9) as usize
    } else if array_size < current / 2 {
        // If the array is much smaller than current threshold,
        // increase the threshold slightly to avoid unnecessary parallelization
        (current as f64 * 1.1) as usize
    } else {
        // Otherwise, keep the current threshold
        current
    };

    // Ensure the new threshold is reasonable
    let tuned = new_threshold.clamp(500, 50_000);

    // Update the global threshold
    set_global_threshold(tuned);

    tuned
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_threshold() {
        // Test with different element costs
        let threshold1 = adaptive_threshold(10000, 1.0);
        let threshold2 = adaptive_threshold(10000, 5.0);

        // Higher element cost should result in lower threshold
        assert!(threshold2 <= threshold1);

        // Test with different array sizes
        let threshold3 = adaptive_threshold(10000, 1.0);
        let threshold4 = adaptive_threshold(1000000, 1.0);

        // Larger array should result in lower threshold
        assert!(threshold4 <= threshold3);
    }

    #[test]
    fn test_global_threshold() {
        // Set the global threshold
        set_global_threshold(5000);

        // Get the global threshold
        let threshold = get_global_threshold();

        // Should be the value we set
        assert_eq!(threshold, 5000);
    }
}
