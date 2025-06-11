//! Parallel scheduling optimization
//!
//! This module provides functionality for optimizing how work is scheduled
//! across threads in parallel computations.

/// Scheduling strategies for parallel computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// Static scheduling - equal distribution of work
    Static,
    /// Dynamic scheduling - work is distributed dynamically at runtime
    Dynamic,
    /// Guided scheduling - chunk size decreases over time
    Guided,
    /// Adaptive scheduling - automatically chooses the best strategy
    Adaptive,
    /// Work-stealing scheduling - threads steal work from others when idle
    WorkStealing,
}

/// Optimize the scheduling of parallel computation
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `element_cost` - The computational cost per element
/// * `strategy` - The scheduling strategy to use
/// * `num_threads` - The total number of threads available
///
/// # Returns
///
/// The optimal number of threads to use
pub fn optimize_scheduling(
    array_size: usize,
    element_cost: f64,
    strategy: SchedulingStrategy,
    num_threads: usize,
) -> usize {
    // Choose the scheduling strategy
    match strategy {
        SchedulingStrategy::Static => static_scheduling(array_size, element_cost, num_threads),
        SchedulingStrategy::Dynamic => dynamic_scheduling(array_size, element_cost, num_threads),
        SchedulingStrategy::Guided => guided_scheduling(array_size, element_cost, num_threads),
        SchedulingStrategy::WorkStealing => {
            work_stealing_scheduler(array_size, element_cost, num_threads)
        }
        SchedulingStrategy::Adaptive => {
            // Choose the best strategy based on workload characteristics
            if array_size < 10_000 || element_cost < 0.5 {
                // For small arrays or cheap elements, static scheduling is often best
                static_scheduling(array_size, element_cost, num_threads)
            } else if element_cost > 5.0 || array_size > 1_000_000 {
                // For expensive elements or very large arrays, work stealing is often best
                work_stealing_scheduler(array_size, element_cost, num_threads)
            } else if element_cost_varies(element_cost) {
                // If element cost varies significantly, dynamic scheduling is often best
                dynamic_scheduling(array_size, element_cost, num_threads)
            } else {
                // For other cases, guided scheduling is a good default
                guided_scheduling(array_size, element_cost, num_threads)
            }
        }
    }
}

/// Calculate the optimal number of threads for static scheduling
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `element_cost` - The computational cost per element
/// * `num_threads` - The total number of threads available
///
/// # Returns
///
/// The optimal number of threads to use
fn static_scheduling(array_size: usize, element_cost: f64, num_threads: usize) -> usize {
    // For static scheduling, we want to ensure each thread has enough work
    let min_elements_per_thread = 1000;
    let optimal_threads = (array_size / min_elements_per_thread).max(1);

    // Scale based on element cost
    let cost_factor = (element_cost / 2.0).clamp(0.5, 2.0);
    let scaled_threads = (optimal_threads as f64 * cost_factor) as usize;

    // Return the minimum of the scaled optimal threads and available threads
    scaled_threads.min(num_threads)
}

/// Calculate the optimal number of threads for dynamic scheduling
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `element_cost` - The computational cost per element
/// * `num_threads` - The total number of threads available
///
/// # Returns
///
/// The optimal number of threads to use
fn dynamic_scheduling(array_size: usize, element_cost: f64, num_threads: usize) -> usize {
    // For dynamic scheduling, we can use more threads as scheduling overhead is less critical
    let min_elements_per_thread = 500;
    let optimal_threads = (array_size / min_elements_per_thread).max(1);

    // Scale based on element cost
    let cost_factor = (element_cost / 1.5).clamp(0.75, 3.0);
    let scaled_threads = (optimal_threads as f64 * cost_factor) as usize;

    // Return the minimum of the scaled optimal threads and available threads
    scaled_threads.min(num_threads)
}

/// Calculate the optimal number of threads for guided scheduling
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `element_cost` - The computational cost per element
/// * `num_threads` - The total number of threads available
///
/// # Returns
///
/// The optimal number of threads to use
fn guided_scheduling(array_size: usize, element_cost: f64, num_threads: usize) -> usize {
    // For guided scheduling, we can use slightly more threads than static
    let min_elements_per_thread = 750;
    let optimal_threads = (array_size / min_elements_per_thread).max(1);

    // Scale based on element cost
    let cost_factor = (element_cost / 1.75).clamp(0.6, 2.5);
    let scaled_threads = (optimal_threads as f64 * cost_factor) as usize;

    // Return the minimum of the scaled optimal threads and available threads
    scaled_threads.min(num_threads)
}

/// Optimize thread count for work-stealing scheduling
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `element_cost` - The computational cost per element
/// * `num_threads` - The total number of threads available
///
/// # Returns
///
/// The optimal number of threads to use
pub fn work_stealing_scheduler(array_size: usize, element_cost: f64, num_threads: usize) -> usize {
    // For work stealing, we can use all available threads for large workloads
    if array_size > 100_000 && element_cost > 1.0 {
        return num_threads;
    }

    // For smaller workloads, scale threads based on workload size
    let min_elements_per_thread = 250;
    let optimal_threads = (array_size / min_elements_per_thread).max(1);

    // Scale based on element cost
    let cost_factor = element_cost.clamp(0.25, 4.0);
    let scaled_threads = (optimal_threads as f64 * cost_factor) as usize;

    // Return the minimum of the scaled optimal threads and available threads
    scaled_threads.min(num_threads)
}

/// Determine if the element processing cost varies significantly
///
/// This is a placeholder for a more sophisticated function that would
/// analyze the workload to determine if element costs vary.
///
/// # Arguments
///
/// * `element_cost` - The average computational cost per element
///
/// # Returns
///
/// True if element costs vary significantly, false otherwise
fn element_cost_varies(element_cost: f64) -> bool {
    // In a real implementation, this would analyze the workload
    // For now, assume costs vary for medium-cost elements
    element_cost > 1.0 && element_cost < 5.0
}

/// Calculate the optimal chunk size for the given scheduling strategy
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `strategy` - The scheduling strategy to use
///
/// # Returns
///
/// The optimal chunk size
pub fn calculate_chunk_size(array_size: usize, strategy: SchedulingStrategy) -> usize {
    match strategy {
        SchedulingStrategy::Static => {
            // For static scheduling, larger chunks reduce overhead
            let num_threads = rayon::current_num_threads();
            (array_size / num_threads).max(1)
        }
        SchedulingStrategy::Dynamic => {
            // For dynamic scheduling, smaller chunks improve load balancing
            // but very small chunks increase overhead
            (array_size / 100).clamp(64, 1024)
        }
        SchedulingStrategy::Guided => {
            // For guided scheduling, this is a starting chunk size
            // that will decrease over time
            (array_size / 10).clamp(128, 4096)
        }
        SchedulingStrategy::WorkStealing => {
            // For work stealing, slightly larger chunks reduce stealing overhead
            (array_size / 50).clamp(128, 2048)
        }
        SchedulingStrategy::Adaptive => {
            // Choose based on array size
            if array_size < 10_000 {
                // For small arrays, larger chunks
                (array_size / 8).max(64)
            } else if array_size > 1_000_000 {
                // For very large arrays, moderate chunks
                1024
            } else {
                // For medium arrays, balanced chunks
                512
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_stealing_scheduler() {
        // Small workload should use fewer threads
        let small_threads = work_stealing_scheduler(1000, 0.5, 12);
        assert!(small_threads < 12);

        // Large workload should use all available threads
        let large_threads = work_stealing_scheduler(1_000_000, 2.0, 12);
        assert_eq!(large_threads, 12);

        // Medium workload should scale with cost
        let medium_low_cost = work_stealing_scheduler(50_000, 0.5, 12);
        let medium_high_cost = work_stealing_scheduler(50_000, 2.0, 12);
        assert!(medium_high_cost >= medium_low_cost);
    }

    #[test]
    fn test_optimize_scheduling() {
        let num_threads = 12;

        // Compare strategies
        let static_threads =
            optimize_scheduling(100_000, 1.0, SchedulingStrategy::Static, num_threads);
        let dynamic_threads =
            optimize_scheduling(100_000, 1.0, SchedulingStrategy::Dynamic, num_threads);
        let guided_threads =
            optimize_scheduling(100_000, 1.0, SchedulingStrategy::Guided, num_threads);

        // Dynamic should use more threads than static for this workload
        assert!(dynamic_threads >= static_threads);

        // Guided should be somewhere in between
        assert!(guided_threads >= static_threads);
    }

    #[test]
    fn test_chunk_size() {
        // Static should have largest chunks
        let static_size = calculate_chunk_size(10_000, SchedulingStrategy::Static);
        let dynamic_size = calculate_chunk_size(10_000, SchedulingStrategy::Dynamic);

        assert!(static_size > dynamic_size);

        // Very small arrays should have reasonably sized chunks
        let small_static = calculate_chunk_size(100, SchedulingStrategy::Static);
        assert!(small_static >= 1);

        // Very large arrays should have reasonable chunks for dynamic
        let large_dynamic = calculate_chunk_size(10_000_000, SchedulingStrategy::Dynamic);
        assert!(large_dynamic <= 1024);
    }
}
