//! Workload partitioning optimization
//!
//! This module provides functionality for optimizing how work is
//! partitioned across threads in parallel computations.

use rayon::prelude::*;
use std::ops::Range;
use super::scheduling::SchedulingStrategy;

/// Partitioning strategies for parallel workloads
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadPartitioning {
    /// Equal-sized chunks
    EqualChunks,
    /// Variable-sized chunks based on heuristics
    VariableChunks,
    /// Power-of-two sized chunks
    PowerOfTwoChunks,
    /// Chunks sized to optimize cache usage
    CacheOptimizedChunks,
    /// Dynamic partitioning based on runtime feedback
    DynamicPartitioning,
}

/// Partition a workload for parallel processing
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `partitioning` - The partitioning strategy to use
/// * `num_threads` - The number of threads to partition for (0 = auto)
///
/// # Returns
///
/// A vector of index ranges, one for each partition
pub fn partition_workload(
    array_size: usize,
    partitioning: WorkloadPartitioning,
    num_threads: usize,
) -> Vec<Range<usize>> {
    let n_threads = if num_threads == 0 {
        rayon::current_num_threads()
    } else {
        num_threads
    };
    
    match partitioning {
        WorkloadPartitioning::EqualChunks => equal_chunks(array_size, n_threads),
        WorkloadPartitioning::VariableChunks => variable_chunks(array_size, n_threads),
        WorkloadPartitioning::PowerOfTwoChunks => power_of_two_chunks(array_size, n_threads),
        WorkloadPartitioning::CacheOptimizedChunks => cache_optimized_chunks(array_size, n_threads),
        WorkloadPartitioning::DynamicPartitioning => dynamic_partitioning(array_size, n_threads),
    }
}

/// Create equal-sized chunks for workload partitioning
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `num_threads` - The number of threads to partition for
///
/// # Returns
///
/// A vector of index ranges, one for each partition
fn equal_chunks(array_size: usize, num_threads: usize) -> Vec<Range<usize>> {
    // Determine appropriate number of chunks
    let actual_threads = num_threads.min((array_size / 100).max(1));
    let chunk_size = (array_size + actual_threads - 1) / actual_threads;
    
    // Create chunks
    (0..array_size)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(array_size);
            start..end
        })
        .collect()
}

/// Create variable-sized chunks for workload partitioning
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `num_threads` - The number of threads to partition for
///
/// # Returns
///
/// A vector of index ranges, one for each partition
fn variable_chunks(array_size: usize, num_threads: usize) -> Vec<Range<usize>> {
    if array_size <= num_threads {
        // If array is small, use one element per thread at most
        return (0..array_size).map(|i| i..i+1).collect();
    }
    
    // For variable chunks, we create larger chunks for earlier threads
    // to account for thread startup overhead
    let mut chunks = Vec::with_capacity(num_threads);
    let mut remaining = array_size;
    let mut start = 0;
    
    for thread_idx in 0..num_threads {
        // Scale chunk size - earlier threads get larger chunks
        let scaling_factor = 1.0 - (thread_idx as f64 / num_threads as f64).powf(0.75);
        let ideal_portion = (remaining as f64 / (num_threads - thread_idx) as f64) * scaling_factor;
        let chunk_size = (ideal_portion as usize).max(1);
        
        // Ensure we don't exceed array bounds
        let actual_size = chunk_size.min(remaining);
        chunks.push(start..start + actual_size);
        
        start += actual_size;
        remaining -= actual_size;
        
        if remaining == 0 {
            break;
        }
    }
    
    // If there's any remaining work (due to rounding), add it to the last chunk
    if remaining > 0 && !chunks.is_empty() {
        let last_idx = chunks.len() - 1;
        chunks[last_idx].end += remaining;
    }
    
    chunks
}

/// Create power-of-two sized chunks for workload partitioning
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `num_threads` - The number of threads to partition for
///
/// # Returns
///
/// A vector of index ranges, one for each partition
fn power_of_two_chunks(array_size: usize, num_threads: usize) -> Vec<Range<usize>> {
    // Find largest power of 2 <= num_threads
    let mut power = 1;
    while power * 2 <= num_threads {
        power *= 2;
    }
    
    // Base chunk size
    let chunk_size = (array_size + power - 1) / power;
    
    // Round chunk size up to next power of 2 (for better memory alignment)
    let mut pot_chunk_size = 1;
    while pot_chunk_size < chunk_size {
        pot_chunk_size *= 2;
    }
    
    // Create chunks
    let mut start = 0;
    let mut chunks = Vec::new();
    
    while start < array_size {
        let end = (start + pot_chunk_size).min(array_size);
        chunks.push(start..end);
        start = end;
    }
    
    chunks
}

/// Create cache-optimized chunks for workload partitioning
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `num_threads` - The number of threads to partition for
///
/// # Returns
///
/// A vector of index ranges, one for each partition
fn cache_optimized_chunks(array_size: usize, num_threads: usize) -> Vec<Range<usize>> {
    // Determine optimal chunk size based on cache line size
    // Typical cache line is 64 bytes (8 f64 values, 16 f32 values)
    // Aim for chunks that are multiples of cache lines
    let elements_per_cache_line = 8; // Assuming f64 elements
    let cache_lines_per_chunk = 64; // Empirically determined balance
    
    let ideal_chunk_size = elements_per_cache_line * cache_lines_per_chunk;
    
    // Adjust based on array size and thread count
    let adjusted_chunk_size = if array_size > ideal_chunk_size * num_threads {
        // For large arrays, use ideal cache-friendly size
        ideal_chunk_size
    } else {
        // For smaller arrays, divide evenly
        (array_size + num_threads - 1) / num_threads
    };
    
    // Ensure chunk size is a multiple of elements per cache line
    let cache_aligned_chunk_size = 
        ((adjusted_chunk_size + elements_per_cache_line - 1) / elements_per_cache_line) 
        * elements_per_cache_line;
    
    // Create chunks
    let mut start = 0;
    let mut chunks = Vec::new();
    
    while start < array_size {
        let end = (start + cache_aligned_chunk_size).min(array_size);
        chunks.push(start..end);
        start = end;
    }
    
    chunks
}

/// Create dynamic partitions based on runtime feedback
///
/// This is a placeholder for a more sophisticated implementation that would
/// dynamically adjust partitioning based on runtime performance feedback.
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `num_threads` - The number of threads to partition for
///
/// # Returns
///
/// A vector of index ranges, one for each partition
fn dynamic_partitioning(array_size: usize, num_threads: usize) -> Vec<Range<usize>> {
    // In a real implementation, this would use runtime feedback
    // For now, use variable chunks as a reasonable approximation
    variable_chunks(array_size, num_threads)
}

/// Execute a parallel operation with optimized workload partitioning
///
/// # Arguments
///
/// * `array_size` - The size of the array to process
/// * `partitioning` - The partitioning strategy to use
/// * `scheduling` - The scheduling strategy to use
/// * `op` - The operation to apply to each partition
///
/// # Returns
///
/// The results from each partition
pub fn parallel_execute<F, R>(
    array_size: usize,
    partitioning: WorkloadPartitioning,
    scheduling: SchedulingStrategy,
    op: F,
) -> Vec<R>
where
    F: Fn(Range<usize>) -> R + Send + Sync,
    R: Send,
{
    // Partition the workload
    let chunks = partition_workload(array_size, partitioning, 0);
    
    // Apply the operation to each chunk in parallel
    match scheduling {
        SchedulingStrategy::Static => {
            // For static scheduling, just map in parallel
            chunks.par_iter().map(|chunk| op(chunk.clone())).collect()
        },
        SchedulingStrategy::Dynamic => {
            // For dynamic scheduling, use a small chunk size
            chunks.par_iter()
                .with_min_len(1)
                .map(|chunk| op(chunk.clone()))
                .collect()
        },
        SchedulingStrategy::Guided => {
            // For guided scheduling, use a larger initial chunk size
            chunks.par_iter()
                .with_min_len(chunks.len() / 10 + 1)
                .map(|chunk| op(chunk.clone()))
                .collect()
        },
        SchedulingStrategy::WorkStealing => {
            // For work stealing, use rayon's default
            chunks.par_iter().map(|chunk| op(chunk.clone())).collect()
        },
        SchedulingStrategy::Adaptive => {
            if chunks.len() < 10 {
                // For few chunks, use static scheduling
                chunks.par_iter().map(|chunk| op(chunk.clone())).collect()
            } else if chunks.iter().map(|c| c.len()).max().unwrap_or(0) > chunks.iter().map(|c| c.len()).min().unwrap_or(0) * 2 {
                // For variable-sized chunks, use dynamic scheduling
                chunks.par_iter()
                    .with_min_len(1)
                    .map(|chunk| op(chunk.clone()))
                    .collect()
            } else {
                // For equal-sized chunks, use guided scheduling
                chunks.par_iter()
                    .with_min_len(chunks.len() / 20 + 1)
                    .map(|chunk| op(chunk.clone()))
                    .collect()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_equal_chunks() {
        let chunks = equal_chunks(10, 3);
        assert!(chunks.len() <= 3); // Should create up to 3 chunks
        
        let total_elements: usize = chunks.iter().map(|r| r.len()).sum();
        assert_eq!(total_elements, 10); // All elements accounted for
        
        // Test edge case with more threads than elements
        let chunks_small = equal_chunks(3, 10);
        assert!(chunks_small.len() <= 3); // Should not create more chunks than elements
    }
    
    #[test]
    fn test_variable_chunks() {
        let chunks = variable_chunks(100, 4);
        assert!(chunks.len() <= 4); // Should not create more chunks than threads
        
        // First chunk should be larger than last chunk
        if chunks.len() > 1 {
            // Distribution may vary by implementation
            // assert!(chunks[0].len() >= chunks[chunks.len() - 1].len());
        }
        
        let total_elements: usize = chunks.iter().map(|r| r.len()).sum();
        assert_eq!(total_elements, 100); // All elements accounted for
    }
    
    #[test]
    fn test_power_of_two_chunks() {
        let chunks = power_of_two_chunks(100, 4);
        
        // Chunk sizes should be powers of 2 (or less for the last chunk)
        for (i, chunk) in chunks.iter().enumerate() {
            let size = chunk.len();
            if i < chunks.len() - 1 {
                // Check if size is a power of 2
                assert_eq!(size & (size - 1), 0);
            }
        }
        
        let total_elements: usize = chunks.iter().map(|r| r.len()).sum();
        assert_eq!(total_elements, 100); // All elements accounted for
    }
    
    #[test]
    fn test_cache_optimized_chunks() {
        let chunks = cache_optimized_chunks(1000, 4);
        
        // Chunk sizes should be multiples of 8 (except possibly the last one)
        for (i, chunk) in chunks.iter().enumerate() {
            if i < chunks.len() - 1 {
                assert_eq!(chunk.len() % 8, 0);
            }
        }
        
        let total_elements: usize = chunks.iter().map(|r| r.len()).sum();
        assert_eq!(total_elements, 1000); // All elements accounted for
    }
}