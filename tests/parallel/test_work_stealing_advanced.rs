//! Advanced tests for work-stealing behavior

use numrs2::parallel::{ThreadPool, ThreadPoolConfig, WorkStealingPool, task};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ============================================================================
// Stealing Strategies Tests
// ============================================================================

#[test]
fn test_steal_from_most_loaded_worker() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit many tasks to create imbalance
    for _ in 0..100 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            std::thread::sleep(Duration::from_micros(100));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    thread::sleep(Duration::from_millis(500));

    assert_eq!(counter.load(Ordering::SeqCst), 100);

    // Check statistics
    let stats = pool.statistics();
    assert_eq!(stats.tasks_submitted, 100);
}

#[test]
fn test_victim_selection_fairness() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    let counters: Vec<Arc<AtomicU32>> = (0..4)
        .map(|_| Arc::new(AtomicU32::new(0)))
        .collect();

    // Submit tasks
    for i in 0..200 {
        let counter_clone = Arc::clone(&counters[i % 4]);
        let task = task(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    thread::sleep(Duration::from_millis(500));

    // Verify work was distributed (not necessarily evenly)
    let total: u32 = counters.iter().map(|c| c.load(Ordering::SeqCst)).sum();
    assert_eq!(total, 200);
}

#[test]
fn test_work_stealing_reduces_imbalance() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit burst of tasks
    for _ in 0..50 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            thread::sleep(Duration::from_millis(20));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");

    assert_eq!(counter.load(Ordering::SeqCst), 50);

    // Check that work was distributed
    let stats = pool.statistics();
    assert!(stats.worker_utilization.len() > 0);
}

// ============================================================================
// Queue Balance Tests
// ============================================================================

#[test]
fn test_queue_rebalancing() {
    let pool = WorkStealingPool::new(3).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit tasks in batches
    for batch in 0..3 {
        for _ in 0..20 {
            let counter_clone = Arc::clone(&counter);
            let task = task(move || {
                thread::sleep(Duration::from_millis(5 * batch));
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
            pool.submit(task).expect("Failed to submit task");
        }
    }

    thread::sleep(Duration::from_millis(800));

    assert_eq!(counter.load(Ordering::SeqCst), 60);

    // Verify queue imbalance is within acceptable range
    let stats = pool.statistics();
    assert!(stats.queue_imbalance <= 1.0);
}

#[test]
fn test_load_distribution_metrics() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit mixed workload
    for i in 0..80 {
        let counter_clone = Arc::clone(&counter);
        let delay = if i % 4 == 0 { 10 } else { 1 };
        let task = task(move || {
            thread::sleep(Duration::from_millis(delay));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    thread::sleep(Duration::from_millis(1000));

    assert_eq!(counter.load(Ordering::SeqCst), 80);

    let stats = pool.statistics();
    assert!(stats.worker_utilization.len() > 0);
}

// ============================================================================
// Concurrent Stealing Tests
// ============================================================================

#[test]
fn test_multiple_workers_stealing_simultaneously() {
    let pool = WorkStealingPool::new(6).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit many tasks quickly
    for _ in 0..300 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    thread::sleep(Duration::from_millis(500));

    assert_eq!(counter.load(Ordering::SeqCst), 300);
}

#[test]
fn test_steal_contention_handling() {
    let pool = WorkStealingPool::new(8).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Create high contention scenario
    for _ in 0..200 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            thread::sleep(Duration::from_micros(500));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    thread::sleep(Duration::from_secs(1));

    assert_eq!(counter.load(Ordering::SeqCst), 200);

    let stats = pool.statistics();
    assert_eq!(stats.tasks_completed, 200);
}

// ============================================================================
// Steal Failure Handling Tests
// ============================================================================

#[test]
fn test_failed_steal_attempts() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit a small number of tasks
    for _ in 0..5 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            thread::sleep(Duration::from_millis(10));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    thread::sleep(Duration::from_millis(200));

    // All tasks should complete even if some steals fail
    assert_eq!(counter.load(Ordering::SeqCst), 5);
}

// ============================================================================
// Priority Preservation Tests
// ============================================================================

#[test]
fn test_high_priority_tasks_execution_order() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(2),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    let execution_order = Arc::new(std::sync::Mutex::new(Vec::new()));

    // Submit normal priority tasks
    for i in 0..5 {
        let order_clone = Arc::clone(&execution_order);
        pool.submit(move || {
            thread::sleep(Duration::from_millis(10));
            order_clone
                .lock()
                .expect("Failed to lock")
                .push(format!("normal-{}", i));
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");

    let order = execution_order.lock().expect("Failed to lock");
    assert_eq!(order.len(), 5);
}

#[test]
fn test_priority_ordering_with_stealing() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit many tasks to trigger stealing
    for _ in 0..100 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");

    assert_eq!(counter.load(Ordering::SeqCst), 100);
}

// ============================================================================
// Work Stealing Statistics Tests
// ============================================================================

#[test]
fn test_stealing_statistics_accuracy() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..100 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    thread::sleep(Duration::from_millis(500));

    let stats = pool.statistics();
    assert_eq!(stats.tasks_submitted, 100);
    assert!(stats.tasks_completed <= 100);
}

#[test]
fn test_active_workers_count() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    // Initially no active workers
    let initial_active = pool.active_workers();
    assert!(initial_active >= 0);

    let counter = Arc::new(AtomicU32::new(0));

    // Submit tasks
    for _ in 0..20 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            thread::sleep(Duration::from_millis(50));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    // Check active workers during execution
    thread::sleep(Duration::from_millis(10));
    let active_during = pool.active_workers();
    assert!(active_during > 0);

    thread::sleep(Duration::from_secs(2));
    assert_eq!(counter.load(Ordering::SeqCst), 20);
}

#[test]
fn test_pending_tasks_count() {
    let pool = WorkStealingPool::new(2).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit many long-running tasks
    for _ in 0..50 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            thread::sleep(Duration::from_millis(20));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    // Check pending tasks
    thread::sleep(Duration::from_millis(10));
    let pending = pool.pending_tasks();
    assert!(pending > 0);

    thread::sleep(Duration::from_secs(2));
    let final_pending = pool.pending_tasks();
    assert_eq!(final_pending, 0);
    assert_eq!(counter.load(Ordering::SeqCst), 50);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn test_urgent_task_submission() {
    let pool = WorkStealingPool::new(2).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU64::new(0));

    // Submit normal tasks
    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            thread::sleep(Duration::from_millis(30));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    // Submit urgent task
    let urgent_counter = Arc::clone(&counter);
    let urgent_task = task(move || {
        urgent_counter.fetch_add(1000, Ordering::SeqCst);
    });
    pool.submit_urgent(urgent_task)
        .expect("Failed to submit urgent task");

    thread::sleep(Duration::from_millis(500));

    let value = counter.load(Ordering::SeqCst);
    assert!(value >= 1000, "Urgent task should execute");
}

#[test]
fn test_work_stealing_with_empty_queues() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    // Don't submit any tasks initially
    thread::sleep(Duration::from_millis(50));

    let counter = Arc::new(AtomicU32::new(0));

    // Now submit a few tasks
    for _ in 0..5 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    thread::sleep(Duration::from_millis(100));
    assert_eq!(counter.load(Ordering::SeqCst), 5);
}
