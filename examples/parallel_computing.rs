//! Comprehensive demonstration of NumRS2 parallel computing capabilities
//!
//! This example demonstrates:
//! - Thread pool with work-stealing
//! - Priority-based task scheduling
//! - Parallel algorithms (sort, scan, reduce, map-reduce)
//! - NUMA-aware operations and memory locality
//! - Monte Carlo simulations
//! - Pipeline processing
//! - Load balancing strategies
//! - Performance comparisons with scaling analysis

use numrs2::parallel::{
    BalancingStrategy, LoadBalancer, ParallelArrayOps, ParallelConfig, ParallelFFT,
    ParallelMatrixOps, ParallelPipeline, ParallelScheduler, SchedulerConfig, TaskPriority,
    ThreadPool, ThreadPoolConfig,
};
use scirs2_core::parallel_ops::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    ParallelSlice, ParallelSliceMut,
};
use scirs2_core::random::thread_rng;
use scirs2_core::Complex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("====================================");
    println!("NumRS2 Parallel Computing Demo");
    println!("====================================\n");

    demo_thread_pool();
    demo_work_stealing();
    demo_priority_scheduling();
    demo_parallel_algorithms();
    demo_numa_aware_operations();
    demo_monte_carlo_simulations();
    demo_parallel_pipeline();
    demo_distributed_array_operations();
    demo_load_balancing();
    demo_performance_comparison();
    demo_thread_scaling_analysis();

    println!("\n====================================");
    println!("Demo completed successfully!");
    println!("====================================");
}

fn demo_thread_pool() {
    println!("1. Thread Pool with Work-Stealing");
    println!("-----------------------------------");

    let config = ThreadPoolConfig {
        num_threads: Some(4),
        enable_thread_pinning: false,
        adaptive_threads: false,
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit 100 tasks
    for i in 0..100 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            if i % 20 == 0 {
                println!("  Task {} executing", i);
            }
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");

    let stats = pool.statistics();
    println!("\nThread Pool Statistics:");
    println!("  Tasks submitted: {}", stats.tasks_submitted);
    println!("  Tasks completed: {}", stats.tasks_completed);
    println!("  Active threads: {}", stats.active_threads);
    println!("  Worker utilization: {:?}", stats.worker_utilization);
    println!();
}

fn demo_work_stealing() {
    println!("2. Work-Stealing in Action");
    println!("-----------------------------------");

    let pool = ThreadPool::new().expect("Failed to create thread pool");
    let counter = Arc::new(AtomicU32::new(0));

    // Submit tasks with varying durations
    for i in 0..50 {
        let counter_clone = Arc::clone(&counter);
        let duration = if i % 5 == 0 { 20 } else { 5 };

        pool.submit(move || {
            std::thread::sleep(std::time::Duration::from_millis(duration));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");

    println!(
        "  Completed {} tasks with work-stealing",
        counter.load(Ordering::SeqCst)
    );
    println!("  Work stealing allows idle threads to steal from busy threads");
    println!();
}

fn demo_priority_scheduling() {
    println!("3. Priority-Based Task Scheduling");
    println!("-----------------------------------");

    let config = SchedulerConfig::optimal_for_cores(2);
    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let execution_order = Arc::new(std::sync::Mutex::new(Vec::new()));

    // Submit tasks with different priorities
    let priorities = vec![
        ("Low priority task", TaskPriority::Low),
        ("Normal priority task", TaskPriority::Normal),
        ("High priority task", TaskPriority::High),
        ("Critical priority task", TaskPriority::Critical),
    ];

    for (name, priority) in priorities {
        let order_clone = Arc::clone(&execution_order);
        let name_str = name.to_string();

        scheduler
            .submit_task(
                move || {
                    order_clone
                        .lock()
                        .expect("Failed to lock order")
                        .push(name_str);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                priority,
                None,
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(std::time::Duration::from_millis(200));

    let order = execution_order.lock().expect("Failed to lock order");
    println!("  Execution order:");
    for (i, task) in order.iter().enumerate() {
        println!("    {}. {}", i + 1, task);
    }
    println!();
}

fn demo_parallel_algorithms() {
    println!("4. Parallel Algorithms");
    println!("-----------------------------------");

    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 100,
        ..Default::default()
    };

    // Parallel sort
    println!("  a) Parallel Sort:");
    let ops = ParallelArrayOps::new(config.clone()).expect("Failed to create parallel ops");
    let mut data = vec![9, 7, 5, 11, 12, 2, 14, 3, 10, 6];
    println!("     Before: {:?}", data);
    ops.parallel_sort(&mut data).expect("Failed to sort");
    println!("     After:  {:?}", data);

    // Parallel reduce
    println!("\n  b) Parallel Reduce (sum):");
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let sum = ops
        .parallel_reduce(&data, 0, |a, b| a + b)
        .expect("Failed to reduce");
    println!("     Sum of {:?} = {}", data, sum);

    // Parallel prefix sum
    println!("\n  c) Parallel Prefix Sum:");
    let data = vec![1, 2, 3, 4, 5];
    let mut result = vec![0; 5];
    ops.parallel_prefix_sum(&data, &mut result)
        .expect("Failed to compute prefix sum");
    println!("     Input:  {:?}", data);
    println!("     Output: {:?}", result);

    // Parallel map-reduce
    println!("\n  d) Parallel Map-Reduce (sum of squares):");
    let data = vec![1, 2, 3, 4, 5];
    let sum_of_squares = ops
        .parallel_map_reduce(&data, |x| x * x, |a, b| a + b, 0)
        .expect("Failed to map-reduce");
    println!("     Input: {:?}", data);
    println!("     Sum of squares: {}", sum_of_squares);

    // Parallel filter
    println!("\n  e) Parallel Filter (even numbers):");
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let evens = ops
        .parallel_filter(&data, |&x| x % 2 == 0)
        .expect("Failed to filter");
    println!("     Input:  {:?}", data);
    println!("     Output: {:?}", evens);

    // Parallel matrix operations
    println!("\n  f) Parallel Matrix Multiplication:");
    let matrix_ops = ParallelMatrixOps::new(config.clone()).expect("Failed to create matrix ops");
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let b = vec![2.0, 0.0, 1.0, 2.0]; // 2x2
    let mut c = vec![0.0; 4]; // 2x2
    matrix_ops
        .parallel_matmul(&a, &b, &mut c, 2, 2, 2)
        .expect("Failed to multiply matrices");
    println!("     A = {:?}", a);
    println!("     B = {:?}", b);
    println!("     C = A * B = {:?}", c);

    // Parallel FFT
    println!("\n  g) Parallel FFT:");
    let fft = ParallelFFT::<f64>::new(config).expect("Failed to create FFT");
    let mut data = vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    println!("     Input:  {:?}", data);
    fft.parallel_fft(&mut data).expect("Failed to compute FFT");
    println!("     FFT:    {:?}", data);

    println!();
}

fn demo_parallel_pipeline() {
    println!("7. Parallel Pipeline Processing");
    println!("-----------------------------------");

    let config = ParallelConfig {
        num_threads: Some(4),
        parallel_threshold: 10,
        ..Default::default()
    };

    let pipeline = ParallelPipeline::<i32>::new(config);

    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // Two-stage pipeline: double then add 1
    let result = pipeline
        .execute(&data, |x| x * 2, |x| x + 1)
        .expect("Failed to execute pipeline");
    println!("  Two-stage (x * 2, then + 1):");
    println!("    Input:  {:?}", data);
    println!("    Output: {:?}", result);

    // Three-stage pipeline: double, add 1, multiply by 3
    let result_3 = pipeline
        .execute_3stage(&data, |x| x * 2, |x| x + 1, |x| x * 3)
        .expect("Failed to execute 3-stage pipeline");
    println!("\n  Three-stage (x * 2, then + 1, then * 3):");
    println!("    Input:  {:?}", data);
    println!("    Output: {:?}", result_3);

    println!();
}

fn demo_load_balancing() {
    println!("11. Dynamic Load Balancing");
    println!("-----------------------------------");

    let strategies = vec![
        BalancingStrategy::RoundRobin,
        BalancingStrategy::LeastLoaded,
        BalancingStrategy::WeightedCapacity,
    ];

    for strategy in strategies {
        println!("  Strategy: {:?}", strategy);
        let balancer = LoadBalancer::new(strategy, 4).expect("Failed to create load balancer");

        // Simulate some load
        for i in 0..4 {
            balancer
                .update_worker_metrics(i, i * 2, 0.5 + i as f64 * 0.1, 0.3)
                .expect("Failed to update metrics");
        }

        let metrics = balancer.current_metrics();
        println!("    Queue lengths: {:?}", metrics.queue_lengths);
        println!("    Load imbalance: {:.3}", metrics.load_imbalance);
        println!("    CPU utilization: {:?}", metrics.cpu_utilization);
        println!();
    }
}

fn demo_numa_aware_operations() {
    println!("5. NUMA-Aware Operations");
    println!("-----------------------------------");

    // NUMA-aware load balancing
    println!("  a) NUMA-Aware Load Balancing:");
    let numa_balancer =
        LoadBalancer::new(BalancingStrategy::NumaAware, 4).expect("Failed to create load balancer");

    // Simulate workload with memory locality considerations
    let data_size = 1_000_000;
    let data: Vec<f64> = (0..data_size).map(|i| i as f64).collect();

    let start = Instant::now();
    let sum: f64 = data
        .par_chunks(data_size / 4)
        .map(|chunk: &[f64]| chunk.iter().sum::<f64>())
        .sum();
    let numa_duration = start.elapsed();

    println!("    Data size: {} elements", data_size);
    println!(
        "    NUMA-aware execution time: {:.6} sec",
        numa_duration.as_secs_f64()
    );
    println!("    Sum computed: {:.2e}", sum);

    // Memory locality optimization
    println!("\n  b) Memory Locality Optimization:");
    let config = ParallelConfig {
        num_threads: Some(4),
        numa_aware: true,
        ..Default::default()
    };
    let matrix_size = 500;
    let matrix_a: Vec<f64> = (0..matrix_size * matrix_size).map(|i| i as f64).collect();
    let matrix_b: Vec<f64> = (0..matrix_size * matrix_size)
        .map(|i| (i % 100) as f64)
        .collect();
    let mut matrix_c = vec![0.0; matrix_size * matrix_size];

    let matrix_ops = ParallelMatrixOps::new(config).expect("Failed to create matrix ops");

    let start = Instant::now();
    matrix_ops
        .parallel_matmul(
            &matrix_a,
            &matrix_b,
            &mut matrix_c,
            matrix_size,
            matrix_size,
            matrix_size,
        )
        .expect("Failed to multiply matrices");
    let matmul_duration = start.elapsed();

    println!("    Matrix size: {}x{}", matrix_size, matrix_size);
    println!(
        "    NUMA-aware matmul time: {:.6} sec",
        matmul_duration.as_secs_f64()
    );
    println!(
        "    Result sample: [{:.2}, {:.2}, ...]",
        matrix_c[0], matrix_c[1]
    );

    let metrics = numa_balancer.current_metrics();
    println!("\n  c) NUMA Metrics:");
    println!("    Load imbalance: {:.3}", metrics.load_imbalance);
    println!("    Work steals: {}", metrics.work_steals);
    println!();
}

fn demo_monte_carlo_simulations() {
    println!("6. Monte Carlo Simulations");
    println!("-----------------------------------");

    // Pi estimation using Monte Carlo
    println!("  a) Pi Estimation:");
    let num_samples = 10_000_000;

    // Sequential version
    let start = Instant::now();
    let mut rng = thread_rng();
    let mut inside_circle = 0u64;
    for _ in 0..num_samples {
        let x: f64 = rng.gen_range(-1.0..1.0);
        let y: f64 = rng.gen_range(-1.0..1.0);
        if x * x + y * y <= 1.0 {
            inside_circle += 1;
        }
    }
    let pi_seq = 4.0 * inside_circle as f64 / num_samples as f64;
    let seq_duration = start.elapsed();

    // Parallel version
    let start = Instant::now();
    let inside_circle_parallel = Arc::new(AtomicU64::new(0));
    let samples_per_thread = num_samples / 4;

    (0..4).into_par_iter().for_each(|_| {
        let mut rng = thread_rng();
        let mut local_inside = 0u64;
        for _ in 0..samples_per_thread {
            let x: f64 = rng.gen_range(-1.0..1.0);
            let y: f64 = rng.gen_range(-1.0..1.0);
            if x * x + y * y <= 1.0 {
                local_inside += 1;
            }
        }
        inside_circle_parallel.fetch_add(local_inside, Ordering::Relaxed);
    });

    let pi_par = 4.0 * inside_circle_parallel.load(Ordering::Relaxed) as f64 / num_samples as f64;
    let par_duration = start.elapsed();

    println!("    Samples: {}", num_samples);
    println!(
        "    Sequential π ≈ {:.6} (time: {:.3}s)",
        pi_seq,
        seq_duration.as_secs_f64()
    );
    println!(
        "    Parallel π ≈ {:.6} (time: {:.3}s)",
        pi_par,
        par_duration.as_secs_f64()
    );
    println!(
        "    Speedup: {:.2}x",
        seq_duration.as_secs_f64() / par_duration.as_secs_f64()
    );
    println!(
        "    Error from π: {:.6}",
        (pi_par - std::f64::consts::PI).abs()
    );

    // Portfolio risk simulation
    println!("\n  b) Portfolio Risk Simulation:");
    let num_simulations = 1_000_000;
    let num_assets = 10;
    let returns = vec![0.08, 0.10, 0.12, 0.07, 0.09, 0.11, 0.08, 0.10, 0.09, 0.11];
    let volatilities = vec![0.15, 0.18, 0.20, 0.12, 0.16, 0.19, 0.14, 0.17, 0.15, 0.18];

    let start = Instant::now();
    let portfolio_values: Vec<f64> = (0..num_simulations)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut portfolio_value = 0.0;
            for i in 0..num_assets {
                let random_return = rng.random::<f64>() * volatilities[i] * 2.0 - volatilities[i];
                portfolio_value += (1.0 + returns[i] + random_return) * 1000.0;
            }
            portfolio_value
        })
        .collect();

    let simulation_duration = start.elapsed();

    let mean_value = portfolio_values.iter().sum::<f64>() / num_simulations as f64;
    let var_95 = {
        let mut sorted = portfolio_values.clone();
        sorted.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[(num_simulations as f64 * 0.05) as usize]
    };

    println!("    Simulations: {}", num_simulations);
    println!("    Assets: {}", num_assets);
    println!("    Mean portfolio value: ${:.2}", mean_value);
    println!("    Value at Risk (95%): ${:.2}", var_95);
    println!(
        "    Simulation time: {:.3}s",
        simulation_duration.as_secs_f64()
    );
    println!();
}

fn demo_distributed_array_operations() {
    println!("8. Distributed Array Operations");
    println!("-----------------------------------");

    // Large-scale array processing
    println!("  a) Large-Scale Element-wise Operations:");
    let size = 5_000_000;
    let array_a: Vec<f64> = (0..size).map(|i| i as f64 * 0.5).collect();
    let array_b: Vec<f64> = (0..size).map(|i| (i % 1000) as f64).collect();

    let start = Instant::now();
    let result: Vec<f64> = array_a
        .par_iter()
        .zip(array_b.par_iter())
        .map(|(a, b): (&f64, &f64)| a.sin() + b.cos())
        .collect();
    let ops_duration = start.elapsed();

    println!("    Array size: {} elements", size);
    println!("    Operation: sin(a) + cos(b)");
    println!("    Time: {:.6} sec", ops_duration.as_secs_f64());
    println!(
        "    Throughput: {:.2} M ops/sec",
        size as f64 / ops_duration.as_secs_f64() / 1e6
    );
    println!(
        "    Result sample: [{:.4}, {:.4}, ...]",
        result[0], result[1]
    );

    // Parallel reduction with complex operations
    println!("\n  b) Parallel Reduction (Standard Deviation):");
    let data: Vec<f64> = (0..1_000_000).map(|i| (i as f64 * 0.1).sin()).collect();

    let start = Instant::now();
    let mean = data.par_iter().sum::<f64>() / data.len() as f64;
    let variance = data
        .par_iter()
        .map(|x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64;
    let std_dev = variance.sqrt();
    let reduction_duration = start.elapsed();

    println!("    Data size: {} elements", data.len());
    println!("    Mean: {:.6}", mean);
    println!("    Std Dev: {:.6}", std_dev);
    println!("    Time: {:.6} sec", reduction_duration.as_secs_f64());

    // Parallel matrix transpose
    println!("\n  c) Parallel Matrix Transpose:");
    let rows = 2000;
    let cols = 3000;
    let matrix: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();

    let start = Instant::now();
    let matrix_ref = &matrix;
    let transposed: Vec<f64> = (0..cols)
        .into_par_iter()
        .flat_map(|col| {
            (0..rows)
                .into_par_iter()
                .map(move |row| matrix_ref[row * cols + col])
        })
        .collect();
    let transpose_duration = start.elapsed();

    println!("    Matrix size: {}x{}", rows, cols);
    println!(
        "    Transposed elements: {} (sample transposed[0] = {:.1})",
        transposed.len(),
        transposed[0]
    );
    println!(
        "    Transpose time: {:.6} sec",
        transpose_duration.as_secs_f64()
    );
    println!(
        "    Memory bandwidth: {:.2} GB/s",
        (rows * cols * 16) as f64 / transpose_duration.as_secs_f64() / 1e9
    );
    println!();
}

fn demo_performance_comparison() {
    println!("9. Performance Comparison: Serial vs Parallel");
    println!("-----------------------------------");

    let sizes = vec![1000, 10000, 100000, 1_000_000];

    println!(
        "  {:>10} | {:>12} | {:>12} | {:>8} | {:>10}",
        "Size", "Sequential", "Parallel", "Speedup", "Efficiency"
    );
    println!("  {}", "-".repeat(70));

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|x| x as f64).collect();

        // Sequential
        let start = Instant::now();
        let sequential_sum: f64 = data.iter().sum();
        let seq_duration = start.elapsed();

        // Parallel
        let config = ParallelConfig {
            num_threads: Some(4),
            parallel_threshold: 100,
            ..Default::default()
        };
        let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

        let start = Instant::now();
        let parallel_sum = ops
            .parallel_reduce(&data, 0.0, |a, b| a + b)
            .expect("Failed to reduce");
        let par_duration = start.elapsed();

        assert!((sequential_sum - parallel_sum).abs() < 1e-6);

        let speedup = seq_duration.as_secs_f64() / par_duration.as_secs_f64();
        let efficiency = speedup / 4.0 * 100.0;

        println!(
            "  {:>10} | {:>9.6}s | {:>9.6}s | {:>7.2}x | {:>9.1}%",
            size,
            seq_duration.as_secs_f64(),
            par_duration.as_secs_f64(),
            speedup,
            efficiency
        );
    }
    println!();
}

fn demo_thread_scaling_analysis() {
    println!("10. Thread Scaling Analysis");
    println!("-----------------------------------");

    let size = 5_000_000;
    let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.5).collect();
    let thread_counts = vec![1, 2, 4, 8];

    println!("  Workload: Sum of {} elements", size);
    println!(
        "\n  {:>8} | {:>12} | {:>8} | {:>10}",
        "Threads", "Time", "Speedup", "Efficiency"
    );
    println!("  {}", "-".repeat(50));

    let baseline_time = {
        let config = ParallelConfig {
            num_threads: Some(1),
            parallel_threshold: 0,
            ..Default::default()
        };
        let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");
        let start = Instant::now();
        let _ = ops
            .parallel_reduce(&data, 0.0, |a, b| a + b)
            .expect("Failed to reduce");
        start.elapsed()
    };

    for num_threads in thread_counts {
        let config = ParallelConfig {
            num_threads: Some(num_threads),
            parallel_threshold: 0,
            ..Default::default()
        };
        let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

        let start = Instant::now();
        let _ = ops
            .parallel_reduce(&data, 0.0, |a, b| a + b)
            .expect("Failed to reduce");
        let duration = start.elapsed();

        let speedup = baseline_time.as_secs_f64() / duration.as_secs_f64();
        let efficiency = speedup / num_threads as f64 * 100.0;

        println!(
            "  {:>8} | {:>9.6}s | {:>7.2}x | {:>9.1}%",
            num_threads,
            duration.as_secs_f64(),
            speedup,
            efficiency
        );
    }

    println!("\n  Analysis:");
    println!("    - Linear scaling would show speedup = thread count");
    println!("    - Efficiency = (speedup / threads) * 100%");
    println!("    - Good parallel code maintains >70% efficiency");
    println!();
}
