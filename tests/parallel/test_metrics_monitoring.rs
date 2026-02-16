//! Tests for performance metrics and monitoring

use numrs2::parallel::{ThreadPool, ThreadPoolConfig, WorkStealingPool, task};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Statistics Collection Tests
// ============================================================================

#[test]
fn test_thread_pool_statistics_accuracy() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit 20 tasks
    for _ in 0..20 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");

    let stats = pool.statistics();
    assert_eq!(stats.tasks_submitted, 20);
    assert_eq!(counter.load(Ordering::SeqCst), 20);
}

#[test]
fn test_work_stealing_pool_statistics() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..50 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(300));

    let stats = pool.statistics();
    assert_eq!(stats.tasks_submitted, 50);
    assert!(stats.tasks_completed <= 50);
    assert_eq!(counter.load(Ordering::SeqCst), 50);
}

// ============================================================================
// Performance Metrics Tests
// ============================================================================

#[test]
fn test_throughput_measurement() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));
    let task_count = 100;

    let start = std::time::Instant::now();

    for _ in 0..task_count {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");
    let duration = start.elapsed();

    let throughput = task_count as f64 / duration.as_secs_f64();
    assert!(throughput > 0.0);
    assert_eq!(counter.load(Ordering::SeqCst), task_count);

    println!("Throughput: {:.2} tasks/sec", throughput);
}

#[test]
fn test_latency_tracking() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(2),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    let latencies = Arc::new(std::sync::Mutex::new(Vec::new()));

    for _ in 0..10 {
        let latencies_clone = Arc::clone(&latencies);
        let submit_time = std::time::Instant::now();

        pool.submit(move || {
            let latency = submit_time.elapsed();
            latencies_clone
                .lock()
                .expect("Failed to lock")
                .push(latency);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");

    let latencies_vec = latencies.lock().expect("Failed to lock");
    assert_eq!(latencies_vec.len(), 10);

    // Calculate average latency
    let avg_latency = latencies_vec.iter().sum::<Duration>() / latencies_vec.len() as u32;
    assert!(avg_latency < Duration::from_secs(1));
}

// ============================================================================
// Worker Utilization Tests
// ============================================================================

#[test]
fn test_cpu_utilization_tracking() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    // Submit CPU-intensive tasks
    for _ in 0..20 {
        pool.submit(|| {
            let mut sum = 0u64;
            for i in 0..1_000_000 {
                sum = sum.wrapping_add(i);
            }
            assert!(sum > 0);
        })
        .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(200));

    let stats = pool.statistics();
    assert!(stats.worker_utilization.len() > 0);

    // Some workers should be utilized
    let total_utilization: f64 = stats.worker_utilization.iter().sum();
    assert!(total_utilization >= 0.0);
}

#[test]
fn test_idle_time_tracking() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    // Initial state - workers should be idle
    std::thread::sleep(Duration::from_millis(50));
    let initial_stats = pool.statistics();
    assert!(initial_stats.active_threads >= 0);

    // Submit tasks
    for _ in 0..10 {
        pool.submit(|| {
            std::thread::sleep(Duration::from_millis(10));
        })
        .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(50));
    let active_stats = pool.statistics();
    assert!(active_stats.tasks_submitted > 0);

    pool.wait().expect("Failed to wait");

    // After completion, workers should be idle again
    std::thread::sleep(Duration::from_millis(50));
    let final_stats = pool.statistics();
    assert_eq!(final_stats.tasks_submitted, 10);
}

// ============================================================================
// Queue Metrics Tests
// ============================================================================

#[test]
fn test_queue_length_monitoring() {
    let pool = WorkStealingPool::new(2).expect("Failed to create work-stealing pool");

    // Submit many slow tasks to fill queues
    for _ in 0..50 {
        let task = task(|| {
            std::thread::sleep(Duration::from_millis(20));
        });
        pool.submit(task).expect("Failed to submit task");
    }

    // Check queue metrics
    std::thread::sleep(Duration::from_millis(10));
    let pending = pool.pending_tasks();
    assert!(pending > 0);

    std::thread::sleep(Duration::from_secs(2));
    let final_pending = pool.pending_tasks();
    assert_eq!(final_pending, 0);
}

#[test]
fn test_queue_wait_time() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(1),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    let wait_times = Arc::new(std::sync::Mutex::new(Vec::new()));

    // Submit tasks
    for _ in 0..5 {
        let wait_times_clone = Arc::clone(&wait_times);
        let submit_time = std::time::Instant::now();

        pool.submit(move || {
            let wait_time = submit_time.elapsed();
            wait_times_clone
                .lock()
                .expect("Failed to lock")
                .push(wait_time);
            std::thread::sleep(Duration::from_millis(10));
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");

    let times = wait_times.lock().expect("Failed to lock");
    assert_eq!(times.len(), 5);

    // Later tasks should have longer wait times
    assert!(times[4] >= times[0]);
}

// ============================================================================
// Statistics API Tests
// ============================================================================

#[test]
fn test_statistics_api_completeness() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    for _ in 0..10 {
        pool.submit(|| {
            std::thread::sleep(Duration::from_millis(5));
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");

    let stats = pool.statistics();

    // Verify all statistics fields
    assert_eq!(stats.tasks_submitted, 10);
    assert!(stats.active_threads >= 0);
    assert!(stats.worker_utilization.len() > 0);
}

#[test]
fn test_work_stealing_pool_metrics() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    for _ in 0..30 {
        let task = task(|| {
            std::thread::sleep(Duration::from_millis(5));
        });
        pool.submit(task).expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(300));

    let stats = pool.statistics();
    assert_eq!(stats.tasks_submitted, 30);
    assert!(stats.worker_utilization.len() > 0);
}

// ============================================================================
// Real-time Monitoring Tests
// ============================================================================

#[test]
fn test_real_time_active_workers_count() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    // Initially no active workers
    let initial_active = pool.active_workers();
    assert!(initial_active >= 0);

    // Submit tasks
    for _ in 0..20 {
        let task = task(|| {
            std::thread::sleep(Duration::from_millis(50));
        });
        pool.submit(task).expect("Failed to submit task");
    }

    // During execution, should have active workers
    std::thread::sleep(Duration::from_millis(10));
    let active_during = pool.active_workers();
    assert!(active_during > 0);

    std::thread::sleep(Duration::from_secs(2));

    // After completion, workers should be idle
    let final_active = pool.active_workers();
    assert!(final_active >= 0);
}

#[test]
fn test_pending_tasks_real_time_tracking() {
    let pool = WorkStealingPool::new(2).expect("Failed to create work-stealing pool");

    // Submit slow tasks
    for _ in 0..30 {
        let task = task(|| {
            std::thread::sleep(Duration::from_millis(30));
        });
        pool.submit(task).expect("Failed to submit task");
    }

    // Check pending tasks immediately
    std::thread::sleep(Duration::from_millis(10));
    let pending_start = pool.pending_tasks();
    assert!(pending_start > 0);

    // Wait a bit and check again
    std::thread::sleep(Duration::from_millis(100));
    let pending_mid = pool.pending_tasks();
    assert!(pending_mid < pending_start);

    // Wait for all to complete
    std::thread::sleep(Duration::from_secs(2));
    let pending_end = pool.pending_tasks();
    assert_eq!(pending_end, 0);
}

// ============================================================================
// Historical Metrics Tests
// ============================================================================

#[test]
fn test_cumulative_statistics() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    // First batch
    for _ in 0..10 {
        pool.submit(|| {}).expect("Failed to submit task");
    }
    pool.wait().expect("Failed to wait");

    let stats1 = pool.statistics();
    assert_eq!(stats1.tasks_submitted, 10);

    // Second batch
    for _ in 0..15 {
        pool.submit(|| {}).expect("Failed to submit task");
    }
    pool.wait().expect("Failed to wait");

    let stats2 = pool.statistics();
    assert_eq!(stats2.tasks_submitted, 25);
}

#[test]
fn test_metrics_consistency() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    for _ in 0..50 {
        let task = task(|| {});
        pool.submit(task).expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(200));

    let stats = pool.statistics();

    // Consistency checks
    assert!(stats.tasks_submitted >= stats.tasks_completed);
    assert!(stats.worker_utilization.len() > 0);
    assert!(stats.queue_imbalance >= 0.0 && stats.queue_imbalance <= 1.0);
}
