//! Tests for scheduler adaptive granularity

use numrs2::parallel::{ParallelScheduler, SchedulerConfig, TaskPriority};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Task Splitting Tests
// ============================================================================

#[test]
fn test_automatic_task_splitting() {
    let config = SchedulerConfig {
        num_threads: 4,
        enable_adaptive_scheduling: true,
        ..SchedulerConfig::optimal_for_cores(4)
    };

    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit a large task that should be split
    let counter_clone = Arc::clone(&counter);
    scheduler
        .submit_task(
            move || {
                for _ in 0..100 {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                }
                numrs2::parallel::scheduler::TaskResult::Success
            },
            TaskPriority::Normal,
            Some(Duration::from_millis(100)), // Long estimated duration
            None,
        )
        .expect("Failed to submit task");

    std::thread::sleep(Duration::from_millis(300));

    assert_eq!(counter.load(Ordering::SeqCst), 100);

    scheduler.shutdown().expect("Failed to shut down scheduler");
}

#[test]
fn test_task_granularity_threshold() {
    let config = SchedulerConfig {
        num_threads: 2,
        enable_adaptive_scheduling: true,
        work_stealing_threshold: 5,
        ..SchedulerConfig::optimal_for_cores(2)
    };

    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit tasks with different estimated costs
    for i in 0..10 {
        let counter_clone = Arc::clone(&counter);
        let cost = Some(Duration::from_millis(i * 10));

        scheduler
            .submit_task(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::Normal,
                cost,
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(500));

    assert_eq!(counter.load(Ordering::SeqCst), 10);

    scheduler.shutdown().expect("Failed to shut down scheduler");
}

// ============================================================================
// Adaptive Chunk Size Tests
// ============================================================================

#[test]
fn test_dynamic_chunk_size_calculation() {
    let config = SchedulerConfig {
        num_threads: 4,
        enable_adaptive_scheduling: true,
        ..SchedulerConfig::optimal_for_cores(4)
    };

    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit many small tasks
    for _ in 0..100 {
        let counter_clone = Arc::clone(&counter);
        scheduler
            .submit_task(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::Normal,
                Some(Duration::from_micros(100)),
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(500));

    assert_eq!(counter.load(Ordering::SeqCst), 100);

    scheduler.shutdown().expect("Failed to shut down scheduler");
}

#[test]
fn test_chunk_size_based_on_thread_count() {
    for num_threads in [1, 2, 4, 8] {
        let available = std::thread::available_parallelism().map_or(4, |n| n.get());
        if num_threads > available {
            continue;
        }

        let config = SchedulerConfig {
            num_threads,
            enable_adaptive_scheduling: true,
            ..SchedulerConfig::optimal_for_cores(num_threads)
        };

        let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

        let counter = Arc::new(AtomicU32::new(0));

        for _ in 0..50 {
            let counter_clone = Arc::clone(&counter);
            scheduler
                .submit_task(
                    move || {
                        counter_clone.fetch_add(1, Ordering::SeqCst);
                        numrs2::parallel::scheduler::TaskResult::Success
                    },
                    TaskPriority::Normal,
                    None,
                    None,
                )
                .expect("Failed to submit task");
        }

        std::thread::sleep(Duration::from_millis(500));

        assert_eq!(
            counter.load(Ordering::SeqCst),
            50,
            "Failed for {} threads",
            num_threads
        );

        scheduler.shutdown().expect("Failed to shut down scheduler");
    }
}

// ============================================================================
// Granularity Tuning Tests
// ============================================================================

#[test]
fn test_fine_grained_tasks() {
    let config = SchedulerConfig {
        num_threads: 4,
        enable_adaptive_scheduling: true,
        time_slice_ms: 1, // Very small time slice
        ..SchedulerConfig::optimal_for_cores(4)
    };

    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit many fine-grained tasks
    for _ in 0..200 {
        let counter_clone = Arc::clone(&counter);
        scheduler
            .submit_task(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::Normal,
                Some(Duration::from_micros(10)),
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(800));

    assert_eq!(counter.load(Ordering::SeqCst), 200);

    scheduler.shutdown().expect("Failed to shut down scheduler");
}

#[test]
fn test_coarse_grained_tasks() {
    let config = SchedulerConfig {
        num_threads: 2,
        enable_adaptive_scheduling: true,
        time_slice_ms: 100, // Large time slice
        ..SchedulerConfig::optimal_for_cores(2)
    };

    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit coarse-grained tasks
    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);
        scheduler
            .submit_task(
                move || {
                    std::thread::sleep(Duration::from_millis(20));
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::Normal,
                Some(Duration::from_millis(50)),
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(800));

    assert_eq!(counter.load(Ordering::SeqCst), 10);

    scheduler.shutdown().expect("Failed to shut down scheduler");
}

// ============================================================================
// Task Cost Estimation Tests
// ============================================================================

#[test]
fn test_task_cost_estimation() {
    let config = SchedulerConfig::optimal_for_cores(4);
    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit tasks with accurate cost estimates
    for i in 1..=10 {
        let counter_clone = Arc::clone(&counter);
        let cost = Duration::from_millis(i * 5);

        scheduler
            .submit_task(
                move || {
                    std::thread::sleep(Duration::from_millis(i * 5));
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::Normal,
                Some(cost),
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(600));

    assert_eq!(counter.load(Ordering::SeqCst), 10);

    let stats = scheduler.statistics();
    assert!(stats.average_execution_time > Duration::ZERO);

    scheduler.shutdown().expect("Failed to shut down scheduler");
}

#[test]
fn test_cost_based_scheduling_decisions() {
    let config = SchedulerConfig {
        num_threads: 4,
        enable_adaptive_scheduling: true,
        ..SchedulerConfig::optimal_for_cores(4)
    };

    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit mix of cheap and expensive tasks
    for i in 0..20 {
        let counter_clone = Arc::clone(&counter);
        let cost = if i % 2 == 0 {
            Duration::from_micros(100) // Cheap
        } else {
            Duration::from_millis(10) // Expensive
        };

        scheduler
            .submit_task(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::Normal,
                Some(cost),
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(500));

    assert_eq!(counter.load(Ordering::SeqCst), 20);

    scheduler.shutdown().expect("Failed to shut down scheduler");
}

// ============================================================================
// Adaptive Scheduling Tests
// ============================================================================

#[test]
fn test_adaptive_scheduling_enabled() {
    let config = SchedulerConfig {
        num_threads: 4,
        enable_adaptive_scheduling: true,
        ..SchedulerConfig::optimal_for_cores(4)
    };

    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit variable-cost tasks
    for i in 0..50 {
        let counter_clone = Arc::clone(&counter);
        let cost = Duration::from_millis((i % 10) as u64);

        scheduler
            .submit_task(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::Normal,
                Some(cost),
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(800));

    assert_eq!(counter.load(Ordering::SeqCst), 50);

    let stats = scheduler.statistics();
    assert!(!stats.thread_efficiency.is_empty());

    scheduler.shutdown().expect("Failed to shut down scheduler");
}

#[test]
fn test_adaptive_scheduling_disabled() {
    let config = SchedulerConfig {
        num_threads: 4,
        enable_adaptive_scheduling: false,
        ..SchedulerConfig::optimal_for_cores(4)
    };

    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..30 {
        let counter_clone = Arc::clone(&counter);
        scheduler
            .submit_task(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::Normal,
                None,
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(500));

    assert_eq!(counter.load(Ordering::SeqCst), 30);

    scheduler.shutdown().expect("Failed to shut down scheduler");
}

// ============================================================================
// Configuration Optimization Tests
// ============================================================================

#[test]
fn test_throughput_optimized_config() {
    let config = SchedulerConfig::throughput_optimized(4);
    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..100 {
        let counter_clone = Arc::clone(&counter);
        scheduler
            .submit_task(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::Normal,
                None,
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(500));

    assert_eq!(counter.load(Ordering::SeqCst), 100);

    scheduler.shutdown().expect("Failed to shut down scheduler");
}

#[test]
fn test_latency_optimized_config() {
    let config = SchedulerConfig::latency_optimized(4);
    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..50 {
        let counter_clone = Arc::clone(&counter);
        scheduler
            .submit_task(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::High, // High priority for low latency
                None,
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(500));

    assert_eq!(counter.load(Ordering::SeqCst), 50);

    scheduler.shutdown().expect("Failed to shut down scheduler");
}
