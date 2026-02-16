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

    pool.wait().expect("Failed to wait");
    assert_eq!(counter.load(Ordering::SeqCst), 100);
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

    pool.wait().expect("Failed to wait");
    assert_eq!(counter.load(Ordering::SeqCst), 50);
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

    // Verify pool is created with valid bounds
    let stats = pool.statistics();
    assert!(stats.active_threads >= 1);
    assert!(stats.active_threads <= 8);
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

    pool.wait().expect("Failed to wait");
    assert_eq!(counter.load(Ordering::SeqCst), 55);
}

#[test]
fn test_thread_pool_configuration_validation() {
    // Test with different valid configurations
    let configs = vec![
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

    pool.wait().expect("Failed to wait");
    assert_eq!(counter.load(Ordering::SeqCst), 50);
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

    std::thread::sleep(Duration::from_millis(150));

    let stats = pool.statistics();
    assert_eq!(stats.tasks_submitted, 20);
    assert!(stats.active_threads > 0);
    assert!(stats.worker_utilization.len() > 0);
}
