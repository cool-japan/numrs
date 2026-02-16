//! Tests for adaptive scheduling

use numrs2::parallel::{ParallelScheduler, SchedulerConfig, TaskPriority};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn test_adaptive_task_granularity() {
    let config = SchedulerConfig::optimal_for_cores(2);
    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit tasks with different priorities
    for i in 0..10 {
        let counter_clone = Arc::clone(&counter);
        let priority = if i % 3 == 0 {
            TaskPriority::High
        } else {
            TaskPriority::Normal
        };

        scheduler
            .submit_task(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                priority,
                Some(Duration::from_millis(10)),
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(500));

    assert_eq!(counter.load(Ordering::SeqCst), 10);

    let stats = scheduler.statistics();
    assert_eq!(stats.tasks_submitted, 10);
    assert!(stats.tasks_completed > 0);
}

#[test]
fn test_task_profiling() {
    let config = SchedulerConfig::optimal_for_cores(2);
    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    // Submit tasks with cost estimates
    for i in 0..5 {
        let cost = Some(Duration::from_millis((i * 10) as u64));
        scheduler
            .submit_task(
                || {
                    std::thread::sleep(Duration::from_millis(20));
                    numrs2::parallel::scheduler::TaskResult::Success
                },
                TaskPriority::Normal,
                cost,
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(300));

    let stats = scheduler.statistics();
    assert_eq!(stats.tasks_submitted, 5);
    assert!(stats.average_execution_time > Duration::ZERO);
}

#[test]
fn test_dynamic_load_balancing() {
    let config = SchedulerConfig {
        num_threads: 4,
        enable_adaptive_scheduling: true,
        work_stealing_threshold: 2,
        ..SchedulerConfig::optimal_for_cores(4)
    };
    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit many tasks to trigger load balancing
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

    std::thread::sleep(Duration::from_millis(1000));

    assert_eq!(counter.load(Ordering::SeqCst), 50);
}

#[test]
fn test_scheduler_statistics_tracking() {
    let config = SchedulerConfig::optimal_for_cores(2);
    let scheduler = ParallelScheduler::new(config).expect("Failed to create scheduler");

    // Submit several tasks
    for _ in 0..10 {
        scheduler
            .submit_task(
                || numrs2::parallel::scheduler::TaskResult::Success,
                TaskPriority::Normal,
                None,
                None,
            )
            .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(200));

    let stats = scheduler.statistics();
    assert_eq!(stats.tasks_submitted, 10);
    assert!(stats.thread_efficiency.len() > 0);
    assert!(stats.thread_efficiency.iter().all(|&e| e >= 0.0 && e <= 1.0));
}
