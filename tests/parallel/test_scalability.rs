//! Scalability benchmarks (1, 2, 4, 8, 16 threads)

use numrs2::parallel::{ParallelArrayOps, ParallelConfig, ThreadPool, ThreadPoolConfig};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[test]
fn test_scalability_1_thread() {
    measure_throughput(1);
}

#[test]
fn test_scalability_2_threads() {
    measure_throughput(2);
}

#[test]
fn test_scalability_4_threads() {
    measure_throughput(4);
}

#[test]
fn test_scalability_8_threads() {
    let available = std::thread::available_parallelism().map_or(4, |n| n.get());
    if available >= 8 {
        measure_throughput(8);
    }
}

fn measure_throughput(num_threads: usize) {
    let config = ThreadPoolConfig {
        num_threads: Some(num_threads),
        ..Default::default()
    };
    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));
    let task_count = 1000;

    let start = Instant::now();

    for _ in 0..task_count {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");
    let duration = start.elapsed();

    assert_eq!(counter.load(Ordering::SeqCst), task_count);

    println!(
        "Throughput with {} threads: {:.2} tasks/sec",
        num_threads,
        task_count as f64 / duration.as_secs_f64()
    );
}

#[test]
fn test_parallel_array_scalability() {
    let sizes = vec![1000, 10000, 100000];

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|x| x as f64).collect();

        for &num_threads in &[1, 2, 4] {
            let config = ParallelConfig {
                num_threads: Some(num_threads),
                parallel_threshold: 100,
                ..Default::default()
            };

            let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

            let start = Instant::now();
            let result = ops
                .parallel_reduce(&data, 0.0, |a, b| a + b)
                .expect("Failed to reduce");
            let duration = start.elapsed();

            // Verify correctness
            let expected: f64 = data.iter().sum();
            assert!((result - expected).abs() < 1e-6);

            println!(
                "Array size {}, {} threads: {:.6} sec",
                size,
                num_threads,
                duration.as_secs_f64()
            );
        }
    }
}

#[test]
fn test_parallel_sort_scalability() {
    let sizes = vec![1000, 10000];

    for size in sizes {
        let mut data: Vec<i32> = (0..size).rev().collect(); // Reverse sorted

        for &num_threads in &[1, 2, 4] {
            let config = ParallelConfig {
                num_threads: Some(num_threads),
                parallel_threshold: 100,
                ..Default::default()
            };

            let ops = ParallelArrayOps::new(config).expect("Failed to create parallel ops");

            let start = Instant::now();
            ops.parallel_sort(&mut data).expect("Failed to sort");
            let duration = start.elapsed();

            // Verify correctness
            for i in 1..data.len() {
                assert!(data[i - 1] <= data[i]);
            }

            println!(
                "Sort size {}, {} threads: {:.6} sec",
                size,
                num_threads,
                duration.as_secs_f64()
            );
        }
    }
}

#[test]
fn test_weak_scaling() {
    // Weak scaling: problem size increases with thread count
    for &num_threads in &[1, 2, 4] {
        let problem_size = num_threads * 1000;

        let config = ThreadPoolConfig {
            num_threads: Some(num_threads),
            ..Default::default()
        };
        let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

        let counter = Arc::new(AtomicU32::new(0));

        let start = Instant::now();

        for _ in 0..problem_size {
            let counter_clone = Arc::clone(&counter);
            pool.submit(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .expect("Failed to submit task");
        }

        pool.wait().expect("Failed to wait for tasks");
        let duration = start.elapsed();

        assert_eq!(counter.load(Ordering::SeqCst), problem_size as u32);

        println!(
            "Weak scaling - {} threads, {} tasks: {:.6} sec",
            num_threads,
            problem_size,
            duration.as_secs_f64()
        );
    }
}

#[test]
fn test_strong_scaling() {
    // Strong scaling: fixed problem size, varying thread count
    let problem_size = 10000;

    for &num_threads in &[1, 2, 4] {
        let config = ThreadPoolConfig {
            num_threads: Some(num_threads),
            ..Default::default()
        };
        let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

        let counter = Arc::new(AtomicU32::new(0));

        let start = Instant::now();

        for _ in 0..problem_size {
            let counter_clone = Arc::clone(&counter);
            pool.submit(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .expect("Failed to submit task");
        }

        pool.wait().expect("Failed to wait for tasks");
        let duration = start.elapsed();

        assert_eq!(counter.load(Ordering::SeqCst), problem_size);

        let speedup = if num_threads == 1 {
            1.0
        } else {
            // Compare against single thread baseline
            1.0 / duration.as_secs_f64()
        };

        println!(
            "Strong scaling - {} threads: {:.6} sec (speedup: {:.2}x)",
            num_threads,
            duration.as_secs_f64(),
            speedup
        );
    }
}

#[test]
fn test_parallel_efficiency() {
    let problem_size = 5000;

    for &num_threads in &[1, 2, 4] {
        let config = ThreadPoolConfig {
            num_threads: Some(num_threads),
            ..Default::default()
        };
        let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

        let counter = Arc::new(AtomicU32::new(0));

        let start = Instant::now();

        for _ in 0..problem_size {
            let counter_clone = Arc::clone(&counter);
            pool.submit(move || {
                // Small amount of work per task
                for _ in 0..100 {
                    counter_clone.fetch_add(0, Ordering::Relaxed); // Minimal work
                }
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .expect("Failed to submit task");
        }

        pool.wait().expect("Failed to wait for tasks");
        let duration = start.elapsed();

        assert_eq!(counter.load(Ordering::SeqCst), problem_size);

        let throughput = problem_size as f64 / duration.as_secs_f64();
        let efficiency = throughput / (num_threads as f64 * problem_size as f64);

        println!(
            "Efficiency - {} threads: {:.2}% ({:.0} tasks/sec)",
            num_threads,
            efficiency * 100.0,
            throughput
        );
    }
}
