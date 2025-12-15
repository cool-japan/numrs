use numrs2::parallel_optimize::{
    adaptive_threshold, optimize_parallel_computation, optimize_scheduling, partition_workload,
    ParallelConfig, ParallelizationThreshold, SchedulingStrategy, WorkloadPartitioning,
};
use scirs2_core::parallel_ops::*;
use std::time::Instant;

fn main() {
    println!("NumRS Parallel Processing Optimization Example");
    println!("============================================\n");

    // SECTION 1: Basic parallelization threshold optimization
    println!("1. Parallelization Threshold Optimization");
    println!("-----------------------------------------");

    // Large array for testing
    let size = 2_000_000;
    let array_data: Vec<f64> = (0..size).map(|i| i as f64).collect();

    // Test with different thresholds
    println!("Testing different parallelization thresholds with element cost = 1.0:");

    // Fixed threshold - too high
    let fixed_high = 1_000_000;
    let time_fixed_high =
        benchmark_parallel_sum(&array_data, ParallelizationThreshold::Fixed(fixed_high));
    println!("  Fixed threshold ({}): {:?}", fixed_high, time_fixed_high);

    // Fixed threshold - too low
    let fixed_low = 1_000;
    let time_fixed_low =
        benchmark_parallel_sum(&array_data, ParallelizationThreshold::Fixed(fixed_low));
    println!("  Fixed threshold ({}): {:?}", fixed_low, time_fixed_low);

    // Adaptive threshold
    let adaptive = adaptive_threshold(size, 1.0);
    let time_adaptive = benchmark_parallel_sum(&array_data, ParallelizationThreshold::Adaptive);
    println!("  Adaptive threshold ({}): {:?}", adaptive, time_adaptive);

    // Calculate speedup
    let best_time = time_fixed_low.min(time_adaptive).min(time_fixed_high);
    println!(
        "  Best threshold provides a {:.2}x speedup over worst threshold",
        time_fixed_high
            .max(time_fixed_low)
            .max(time_adaptive)
            .as_secs_f64()
            / best_time.as_secs_f64()
    );

    // SECTION 2: Scheduling strategies
    println!("\n2. Scheduling Strategy Optimization");
    println!("---------------------------------");

    // Test with different scheduling strategies
    println!("Testing different scheduling strategies:");

    let element_cost = 1.0;
    let num_threads = scirs2_core::parallel_ops::num_threads();

    // Static scheduling
    let static_threads =
        optimize_scheduling(size, element_cost, SchedulingStrategy::Static, num_threads);
    let time_static = benchmark_scheduling(&array_data, SchedulingStrategy::Static);
    println!(
        "  Static scheduling (threads={}): {:?}",
        static_threads, time_static
    );

    // Dynamic scheduling
    let dynamic_threads =
        optimize_scheduling(size, element_cost, SchedulingStrategy::Dynamic, num_threads);
    let time_dynamic = benchmark_scheduling(&array_data, SchedulingStrategy::Dynamic);
    println!(
        "  Dynamic scheduling (threads={}): {:?}",
        dynamic_threads, time_dynamic
    );

    // Guided scheduling
    let guided_threads =
        optimize_scheduling(size, element_cost, SchedulingStrategy::Guided, num_threads);
    let time_guided = benchmark_scheduling(&array_data, SchedulingStrategy::Guided);
    println!(
        "  Guided scheduling (threads={}): {:?}",
        guided_threads, time_guided
    );

    // Work stealing scheduling
    let ws_threads = optimize_scheduling(
        size,
        element_cost,
        SchedulingStrategy::WorkStealing,
        num_threads,
    );
    let time_ws = benchmark_scheduling(&array_data, SchedulingStrategy::WorkStealing);
    println!(
        "  Work-stealing scheduling (threads={}): {:?}",
        ws_threads, time_ws
    );

    // Adaptive scheduling
    let adaptive_threads = optimize_scheduling(
        size,
        element_cost,
        SchedulingStrategy::Adaptive,
        num_threads,
    );
    let time_adaptive = benchmark_scheduling(&array_data, SchedulingStrategy::Adaptive);
    println!(
        "  Adaptive scheduling (threads={}): {:?}",
        adaptive_threads, time_adaptive
    );

    // Calculate speedup
    let best_sched_time = time_static
        .min(time_dynamic)
        .min(time_guided)
        .min(time_ws)
        .min(time_adaptive);
    println!(
        "  Best scheduling strategy provides a {:.2}x speedup over worst strategy",
        time_static
            .max(time_dynamic)
            .max(time_guided)
            .max(time_ws)
            .max(time_adaptive)
            .as_secs_f64()
            / best_sched_time.as_secs_f64()
    );

    // SECTION 3: Workload partitioning
    println!("\n3. Workload Partitioning Optimization");
    println!("-----------------------------------");

    // Test with different partitioning strategies
    println!("Testing different workload partitioning strategies:");

    // Equal chunks
    let time_equal = benchmark_partitioning(&array_data, WorkloadPartitioning::EqualChunks);
    println!("  Equal chunks: {:?}", time_equal);

    // Variable chunks
    let time_variable = benchmark_partitioning(&array_data, WorkloadPartitioning::VariableChunks);
    println!("  Variable chunks: {:?}", time_variable);

    // Power-of-two chunks
    let time_pot = benchmark_partitioning(&array_data, WorkloadPartitioning::PowerOfTwoChunks);
    println!("  Power-of-two chunks: {:?}", time_pot);

    // Cache-optimized chunks
    let time_cache =
        benchmark_partitioning(&array_data, WorkloadPartitioning::CacheOptimizedChunks);
    println!("  Cache-optimized chunks: {:?}", time_cache);

    // Dynamic partitioning
    let time_dynamic =
        benchmark_partitioning(&array_data, WorkloadPartitioning::DynamicPartitioning);
    println!("  Dynamic partitioning: {:?}", time_dynamic);

    // Calculate speedup
    let best_part_time = time_equal
        .min(time_variable)
        .min(time_pot)
        .min(time_cache)
        .min(time_dynamic);
    println!(
        "  Best partitioning strategy provides a {:.2}x speedup over worst strategy",
        time_equal
            .max(time_variable)
            .max(time_pot)
            .max(time_cache)
            .max(time_dynamic)
            .as_secs_f64()
            / best_part_time.as_secs_f64()
    );

    // SECTION 4: Combined optimization
    println!("\n4. Combined Parallel Optimization");
    println!("-------------------------------");

    // Create an array for testing
    let size = 5_000_000;
    let element_cost = 2.0;
    let array_data: Vec<f64> = (0..size).map(|i| i as f64).collect();

    // Test with different configurations
    println!("Testing different optimization configurations:");

    // Default configuration
    let time_default = benchmark_with_config(&array_data, ParallelConfig::default());
    println!("  Default configuration: {:?}", time_default);

    // Optimized configuration
    let optimized_config = ParallelConfig::optimized(size, element_cost);
    let time_optimized = benchmark_with_config(&array_data, optimized_config);
    println!("  Optimized configuration: {:?}", time_optimized);
    println!(
        "    Minimum parallel size: {}",
        optimized_config.min_parallel_size
    );
    println!("    Chunk size: {}", optimized_config.chunk_size);
    println!(
        "    Scheduling strategy: {:?}",
        optimized_config.scheduling_strategy
    );

    // Fine-tuned configuration
    let tuned_config = ParallelConfig::default()
        .with_min_size(adaptive_threshold(size, element_cost))
        .with_chunk_size(size / 64)
        .with_scheduling(SchedulingStrategy::WorkStealing);
    let time_tuned = benchmark_with_config(&array_data, tuned_config);
    println!("  Fine-tuned configuration: {:?}", time_tuned);

    // Calculate speedup
    let best_config_time = time_default.min(time_optimized).min(time_tuned);
    println!(
        "  Best configuration provides a {:.2}x speedup over default",
        time_default.as_secs_f64() / best_config_time.as_secs_f64()
    );

    // SECTION 5: Adapting to different workloads
    println!("\n5. Adapting to Different Workloads");
    println!("---------------------------------");

    // Test with different workload characteristics
    println!("Testing adaptation to different workloads:");

    // Small workload with cheap operations
    let small_size = 10_000;
    let _small_data: Vec<f64> = (0..small_size).map(|i| i as f64).collect();
    let optimal_threads_small =
        optimize_parallel_computation(small_size, 0.1, SchedulingStrategy::Adaptive);
    println!("  Small workload (size={}, cost=0.1):", small_size);
    println!("    Optimal threads: {}", optimal_threads_small);
    println!(
        "    Parallel execution recommended: {}",
        optimal_threads_small > 1
    );

    // Medium workload with moderate operations
    let medium_size = 100_000;
    let _medium_data: Vec<f64> = (0..medium_size).map(|i| i as f64).collect();
    let optimal_threads_medium =
        optimize_parallel_computation(medium_size, 1.0, SchedulingStrategy::Adaptive);
    println!("  Medium workload (size={}, cost=1.0):", medium_size);
    println!("    Optimal threads: {}", optimal_threads_medium);
    println!(
        "    Parallel execution recommended: {}",
        optimal_threads_medium > 1
    );

    // Large workload with expensive operations
    let large_size = 1_000_000;
    let _large_data: Vec<f64> = (0..large_size).map(|i| i as f64).collect();
    let optimal_threads_large =
        optimize_parallel_computation(large_size, 5.0, SchedulingStrategy::Adaptive);
    println!("  Large workload (size={}, cost=5.0):", large_size);
    println!("    Optimal threads: {}", optimal_threads_large);
    println!(
        "    Parallel execution recommended: {}",
        optimal_threads_large > 1
    );

    // Very large workload with moderate operations
    let very_large_size = 10_000_000;
    let optimal_threads_very_large =
        optimize_parallel_computation(very_large_size, 0.5, SchedulingStrategy::Adaptive);
    println!(
        "  Very large workload (size={}, cost=0.5):",
        very_large_size
    );
    println!("    Optimal threads: {}", optimal_threads_very_large);
    println!(
        "    Parallel execution recommended: {}",
        optimal_threads_very_large > 1
    );
}

// Benchmark summation with different parallelization thresholds
fn benchmark_parallel_sum(
    data: &[f64],
    threshold_type: ParallelizationThreshold,
) -> std::time::Duration {
    let start = Instant::now();

    let size = data.len();
    let element_cost = 1.0;
    let threshold = match threshold_type {
        ParallelizationThreshold::Fixed(value) => value,
        ParallelizationThreshold::Adaptive => adaptive_threshold(size, element_cost),
        _ => adaptive_threshold(size, element_cost), // Use adaptive as fallback
    };

    let sum = if size <= threshold {
        // Sequential execution
        data.iter()
            .map(|&x| compute_with_cost(x, element_cost))
            .sum::<f64>()
    } else {
        // Parallel execution
        data.par_iter()
            .map(|&x| compute_with_cost(x, element_cost))
            .sum::<f64>()
    };

    let duration = start.elapsed();

    // Prevent compiler from optimizing away the calculation
    if sum < 0.0 {
        println!("Sum is negative (should never happen): {}", sum);
    }

    duration
}

// Benchmark summation with different scheduling strategies
fn benchmark_scheduling(data: &[f64], strategy: SchedulingStrategy) -> std::time::Duration {
    let start = Instant::now();

    let size = data.len();
    let element_cost = 1.0;
    let num_threads = scirs2_core::parallel_ops::num_threads();
    let optimal_threads = optimize_scheduling(size, element_cost, strategy, num_threads);

    // Create a custom thread pool with the optimal number of threads
    let pool = scirs2_core::parallel_ops::ThreadPoolBuilder::new()
        .num_threads(optimal_threads)
        .build()
        .unwrap();

    let sum = pool.install(|| {
        match strategy {
            SchedulingStrategy::Static => {
                // For static scheduling, divide work evenly
                let chunk_size = size.div_ceil(optimal_threads);
                let chunks: Vec<_> = data.chunks(chunk_size).collect();
                chunks
                    .par_iter()
                    .map(|chunk| {
                        chunk
                            .iter()
                            .map(|&x| compute_with_cost(x, element_cost))
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            }
            SchedulingStrategy::Dynamic => {
                // For dynamic scheduling, use smaller chunks
                let chunk_size = 1000;
                let chunks: Vec<_> = data.chunks(chunk_size).collect();
                chunks
                    .par_iter()
                    .map(|chunk| {
                        chunk
                            .iter()
                            .map(|&x| compute_with_cost(x, element_cost))
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            }
            SchedulingStrategy::Guided => {
                // For guided scheduling, start with larger chunks that get smaller
                data.par_iter()
                    .with_min_len(100)
                    .map(|&x| compute_with_cost(x, element_cost))
                    .sum::<f64>()
            }
            SchedulingStrategy::WorkStealing => {
                // For work stealing, use rayon's default
                data.par_iter()
                    .map(|&x| compute_with_cost(x, element_cost))
                    .sum::<f64>()
            }
            SchedulingStrategy::Adaptive => {
                // For adaptive, choose based on array size
                if size < 100_000 {
                    // Small array - static scheduling
                    let chunk_size = size.div_ceil(optimal_threads);
                    let chunks: Vec<_> = data.chunks(chunk_size).collect();
                    chunks
                        .par_iter()
                        .map(|chunk| {
                            chunk
                                .iter()
                                .map(|&x| compute_with_cost(x, element_cost))
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                } else {
                    // Larger array - guided scheduling
                    data.par_iter()
                        .with_min_len(100)
                        .map(|&x| compute_with_cost(x, element_cost))
                        .sum::<f64>()
                }
            }
        }
    });

    let duration = start.elapsed();

    // Prevent compiler from optimizing away the calculation
    if sum < 0.0 {
        println!("Sum is negative (should never happen): {}", sum);
    }

    duration
}

// Benchmark summation with different workload partitioning
fn benchmark_partitioning(data: &[f64], partitioning: WorkloadPartitioning) -> std::time::Duration {
    let start = Instant::now();

    let size = data.len();
    let element_cost = 1.0;

    // Partition the workload
    let partitions = partition_workload(size, partitioning, 0);

    // Process each partition in parallel
    let sum: f64 = partitions
        .par_iter()
        .map(|range| {
            let mut sum = 0.0;
            for i in range.clone() {
                sum += compute_with_cost(data[i], element_cost);
            }
            sum
        })
        .sum::<f64>();

    let duration = start.elapsed();

    // Prevent compiler from optimizing away the calculation
    if sum < 0.0 {
        println!("Sum is negative (should never happen): {}", sum);
    }

    duration
}

// Benchmark summation with different parallel configurations
fn benchmark_with_config(data: &[f64], config: ParallelConfig) -> std::time::Duration {
    let start = Instant::now();

    let size = data.len();
    let element_cost = 2.0;

    let sum = if !config.should_parallelize(size) {
        // Sequential execution
        data.iter()
            .map(|&x| compute_with_cost(x, element_cost))
            .sum::<f64>()
    } else {
        // Parallel execution with custom settings
        let optimal_threads = config.optimal_threads(size, element_cost);

        // Create a custom thread pool with the optimal number of threads
        let pool = scirs2_core::parallel_ops::ThreadPoolBuilder::new()
            .num_threads(optimal_threads)
            .build()
            .unwrap();

        pool.install(|| {
            match config.scheduling_strategy {
                SchedulingStrategy::Static => {
                    // Static scheduling with custom chunk size
                    let chunks: Vec<_> = data.chunks(config.chunk_size).collect();
                    chunks
                        .par_iter()
                        .map(|chunk| {
                            chunk
                                .iter()
                                .map(|&x| compute_with_cost(x, element_cost))
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                }
                _ => {
                    // Other strategies use rayon's builtin scheduling
                    data.par_iter()
                        .with_min_len(config.chunk_size)
                        .map(|&x| compute_with_cost(x, element_cost))
                        .sum::<f64>()
                }
            }
        })
    };

    let duration = start.elapsed();

    // Prevent compiler from optimizing away the calculation
    if sum < 0.0 {
        println!("Sum is negative (should never happen): {}", sum);
    }

    duration
}

// Simulate a computation with a specified cost
fn compute_with_cost(value: f64, cost: f64) -> f64 {
    // Simulate computation by doing more work for higher costs
    let iterations = (cost * 10.0) as usize;
    let mut result = value;

    for i in 0..iterations {
        // Some arbitrary computation that can't be optimized away
        result = result.sin().cos() + (i as f64) * 0.000001;
    }

    result
}
