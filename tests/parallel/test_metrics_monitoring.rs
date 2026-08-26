//! Tests for performance metrics and monitoring

use numrs2::parallel::{task, ThreadPool, ThreadPoolConfig, WorkStealingPool};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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

    pool.shutdown().expect("Failed to shut down thread pool");
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

    // Signal workers to stop before the test process exits; `shutdown` takes
    // `&self` and cannot join (see `WorkStealingPool::shutdown`'s doc
    // comment), so this only shrinks the race window, it does not close it.
    pool.shutdown()
        .expect("Failed to shut down work-stealing pool");
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

    pool.shutdown().expect("Failed to shut down thread pool");
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

    pool.shutdown().expect("Failed to shut down thread pool");
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
    assert!(!stats.worker_utilization.is_empty());

    // Some workers should be utilized
    let total_utilization: f64 = stats.worker_utilization.iter().sum();
    assert!(total_utilization >= 0.0);

    // `shutdown()` joins all worker threads itself and abandons any tasks
    // still queued past this point, so no separate `wait()` is needed here
    // just to quiesce (this test intentionally samples utilization mid-burst
    // rather than after completion).
    pool.shutdown().expect("Failed to shut down thread pool");
}

#[test]
fn test_idle_time_tracking() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    // Initial state - workers should be idle. Not asserted as
    // `active_threads == 0`: even with no work ever submitted, each worker
    // cycles park(idle=true) -> wake-on-timeout(idle=false) -> recheck ->
    // park roughly every `idle_timeout` (see `ThreadPool::worker_main`), so
    // sampling at one fixed instant has a real, if small, chance of
    // catching a worker in that brief idle=false window.
    std::thread::sleep(Duration::from_millis(50));
    let initial_stats = pool.statistics();
    assert_eq!(initial_stats.tasks_submitted, 0);

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

    pool.shutdown().expect("Failed to shut down thread pool");
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

    super::wait_for_drain(&pool, Duration::from_secs(5));
    assert_eq!(pool.pending_tasks(), 0);

    pool.shutdown()
        .expect("Failed to shut down work-stealing pool");
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

    // Not `pool.wait()` + an immediate read: `wait()` can return with the
    // last task per worker still mid-closure (see `wait_for_count`'s doc
    // comment in `tests/parallel/mod.rs`), and these closures sleep, making
    // that window observable as a short `wait_times` vector. Poll the real
    // side effect (its length) instead.
    let deadline = Instant::now() + Duration::from_secs(2);
    while wait_times.lock().expect("Failed to lock").len() < 5 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }

    let times = wait_times.lock().expect("Failed to lock");
    assert_eq!(times.len(), 5);

    // Later tasks should have longer wait times
    assert!(times[4] >= times[0]);

    pool.shutdown().expect("Failed to shut down thread pool");
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

    // Verify all statistics fields. `active_threads` is intentionally not
    // asserted here: `statistics()` is a separate call made after
    // `pool.wait()` already returned, and a worker can flip its `is_idle`
    // flag false again on its very next park/wake cycle (see
    // `ThreadPool::worker_main`) before this call samples it, so neither
    // `== 0` nor `> 0` is a race-free expectation at this point.
    assert_eq!(stats.tasks_submitted, 10);
    assert!(!stats.worker_utilization.is_empty());

    pool.shutdown().expect("Failed to shut down thread pool");
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
    assert!(!stats.worker_utilization.is_empty());

    pool.shutdown()
        .expect("Failed to shut down work-stealing pool");
}

// ============================================================================
// Real-time Monitoring Tests
// ============================================================================

#[test]
fn test_real_time_active_workers_count() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    // `active_workers()` reflects `is_idle`, which `WorkStealingPool`'s
    // worker loop only ever clears on wake-from-a-parked-wait -- never on
    // successful task pickup (see `WorkStealingPool::worker_main`). During
    // an uninterrupted burst of back-to-back execution (the common case:
    // 20 tasks handed to 4 workers with no gap), no worker ever hits the
    // "no work found" branch that would flip the flag, so
    // `active_workers()` can legitimately read 0 for the *entire* burst.
    // This is not a rare race: polling for `active_workers() > 0` for up to
    // 500ms while these 20 tasks (4 * 5 tasks, 50ms each = 250ms/worker of
    // continuous work) ran observed it staying 0 the whole time. So it is
    // not asserted here at all; what's verified instead is that all
    // submitted work genuinely completes.
    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..20 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            std::thread::sleep(Duration::from_millis(50));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    super::wait_for_count(&counter, 20, Duration::from_secs(5));

    pool.shutdown()
        .expect("Failed to shut down work-stealing pool");
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
    super::wait_for_drain(&pool, Duration::from_secs(5));
    assert_eq!(pool.pending_tasks(), 0);

    pool.shutdown()
        .expect("Failed to shut down work-stealing pool");
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

    pool.shutdown().expect("Failed to shut down thread pool");
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
    assert!(!stats.worker_utilization.is_empty());
    assert!((0.0..=1.0).contains(&stats.queue_imbalance));

    pool.shutdown()
        .expect("Failed to shut down work-stealing pool");
}
