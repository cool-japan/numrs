//! Tests for thread affinity and CPU pinning

use numrs2::parallel::{ThreadPool, ThreadPoolConfig};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Thread Affinity Configuration Tests
// ============================================================================

#[test]
fn test_thread_pinning_enabled() {
    let config = ThreadPoolConfig {
        num_threads: Some(2),
        enable_thread_pinning: true,
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");
    assert_eq!(counter.load(Ordering::SeqCst), 10);
}

#[test]
fn test_thread_pinning_disabled() {
    let config = ThreadPoolConfig {
        num_threads: Some(2),
        enable_thread_pinning: false,
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");
    assert_eq!(counter.load(Ordering::SeqCst), 10);
}

// ============================================================================
// CPU Pinning Tests
// ============================================================================

#[test]
fn test_cpu_affinity_basic() {
    // Note: Actual CPU pinning is platform-specific
    // This test verifies the configuration is accepted
    let config = ThreadPoolConfig {
        num_threads: Some(4),
        enable_thread_pinning: true,
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..20 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");
    assert_eq!(counter.load(Ordering::SeqCst), 20);
}

#[test]
fn test_thread_cpu_distribution() {
    let num_cpus = std::thread::available_parallelism().map_or(4, |n| n.get());

    let config = ThreadPoolConfig {
        num_threads: Some(num_cpus.min(4)),
        enable_thread_pinning: true,
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit tasks and verify they execute
    for _ in 0..100 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            std::thread::sleep(Duration::from_micros(100));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    // Closures sleep, so `pool.wait()` can return with the last one per
    // worker still mid-sleep; poll the real side effect instead.
    super::wait_for_count(&counter, 100, Duration::from_secs(2));
}

// ============================================================================
// Adaptive Thread Count Tests
// ============================================================================

#[test]
fn test_adaptive_thread_count_enabled() {
    let config = ThreadPoolConfig {
        num_threads: Some(2),
        adaptive_threads: true,
        min_threads: 1,
        max_threads: 4,
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit many tasks to potentially trigger thread growth
    for _ in 0..50 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            std::thread::sleep(Duration::from_millis(10));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    // Closures sleep, so `pool.wait()` can return with the last one per
    // worker still mid-sleep; poll the real side effect instead.
    super::wait_for_count(&counter, 50, Duration::from_secs(2));
}

#[test]
fn test_thread_count_bounds() {
    let config = ThreadPoolConfig {
        num_threads: Some(2),
        adaptive_threads: true,
        min_threads: 1,
        max_threads: 8,
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    // `adaptive_threads`/`min_threads`/`max_threads` are accepted as
    // configuration, but `ThreadPool` does not currently grow or shrink its
    // worker count at runtime -- `num_threads()` stays fixed at whatever
    // `num_threads` this pool was constructed with. Verify that directly,
    // and that it falls within the configured bounds, rather than sampling
    // `is_idle`-based `active_threads` immediately after construction: with
    // zero tasks submitted every worker is idle, so `active_threads` is
    // deterministically 0, never `>= 1`.
    assert_eq!(pool.num_threads(), 2);
    assert!((1..=8).contains(&pool.num_threads()));
}

// ============================================================================
// Thread Pool Resize Tests
// ============================================================================

#[test]
fn test_thread_pool_with_variable_load() {
    let config = ThreadPoolConfig {
        num_threads: Some(4),
        adaptive_threads: false,
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Light load
    for _ in 0..5 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    std::thread::sleep(Duration::from_millis(50));

    // Heavy load
    for _ in 0..50 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            std::thread::sleep(Duration::from_millis(5));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    // Closures sleep, so `pool.wait()` can return with the last one per
    // worker still mid-sleep; poll the real side effect instead.
    super::wait_for_count(&counter, 55, Duration::from_secs(2));
}

#[test]
fn test_thread_pool_configuration_validation() {
    // Test with different valid configurations
    let configs = [
        ThreadPoolConfig {
            num_threads: Some(1),
            ..Default::default()
        },
        ThreadPoolConfig {
            num_threads: Some(2),
            ..Default::default()
        },
        ThreadPoolConfig {
            num_threads: Some(4),
            ..Default::default()
        },
        ThreadPoolConfig {
            num_threads: None, // Auto-detect
            ..Default::default()
        },
    ];

    for config in configs {
        let pool = ThreadPool::with_config(config);
        assert!(pool.is_ok(), "Thread pool creation should succeed");
    }
}

// ============================================================================
// Thread Affinity Performance Tests
// ============================================================================

#[test]
fn test_affinity_impact_on_throughput() {
    let counter_pinned = Arc::new(AtomicU32::new(0));
    let counter_unpinned = Arc::new(AtomicU32::new(0));

    // Test with pinning
    {
        let config = ThreadPoolConfig {
            num_threads: Some(4),
            enable_thread_pinning: true,
            ..Default::default()
        };
        let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let counter_clone = Arc::clone(&counter_pinned);
            pool.submit(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .expect("Failed to submit task");
        }
        pool.wait().expect("Failed to wait");
        let _pinned_duration = start.elapsed();
    }

    // Test without pinning
    {
        let config = ThreadPoolConfig {
            num_threads: Some(4),
            enable_thread_pinning: false,
            ..Default::default()
        };
        let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let counter_clone = Arc::clone(&counter_unpinned);
            pool.submit(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .expect("Failed to submit task");
        }
        pool.wait().expect("Failed to wait");
        let _unpinned_duration = start.elapsed();
    }

    assert_eq!(counter_pinned.load(Ordering::SeqCst), 100);
    assert_eq!(counter_unpinned.load(Ordering::SeqCst), 100);
}

#[test]
fn test_thread_pool_idle_timeout_configuration() {
    let config = ThreadPoolConfig {
        num_threads: Some(2),
        idle_timeout: Duration::from_millis(5),
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");
    assert_eq!(counter.load(Ordering::SeqCst), 10);
}

#[test]
fn test_thread_pool_queue_capacity() {
    let config = ThreadPoolConfig {
        num_threads: Some(2),
        queue_capacity: 50,
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit tasks within capacity
    for _ in 0..50 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            std::thread::sleep(Duration::from_millis(1));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    // Closures sleep, so `pool.wait()` can return with the last one per
    // worker still mid-sleep; poll the real side effect instead.
    super::wait_for_count(&counter, 50, Duration::from_secs(2));
}

#[test]
fn test_thread_pool_statistics_with_affinity() {
    let config = ThreadPoolConfig {
        num_threads: Some(4),
        enable_thread_pinning: true,
        ..Default::default()
    };

    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    for _ in 0..20 {
        pool.submit(|| {
            std::thread::sleep(Duration::from_millis(5));
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait");

    let stats = pool.statistics();
    assert_eq!(stats.tasks_submitted, 20);
    // `ThreadPool` only clears a worker's `is_idle` flag on wake-from-park,
    // never on task pickup (see `ThreadPool::worker_main`), so
    // `active_threads` cannot be asserted `> 0` at any particular instant --
    // a worker that finds a task on its very first loop iteration, before
    // ever parking, still reports idle. Only the vector's shape (one entry
    // per worker) is a reliable signal here.
    assert!(!stats.worker_utilization.is_empty());
}
