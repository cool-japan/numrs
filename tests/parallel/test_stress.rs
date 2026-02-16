//! Stress tests with high contention

use numrs2::parallel::{ThreadPool, ThreadPoolConfig, WorkStealingPool, task};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_high_task_count() {
    let pool = ThreadPool::new().expect("Failed to create thread pool");
    let counter = Arc::new(AtomicU32::new(0));

    // Submit a large number of tasks
    for _ in 0..1000 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");
    assert_eq!(counter.load(Ordering::SeqCst), 1000);
}

#[test]
fn test_high_contention_shared_state() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(8),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    let shared_counter = Arc::new(AtomicU32::new(0));

    // Many threads contending for the same counter
    for _ in 0..500 {
        let counter_clone = Arc::clone(&shared_counter);
        pool.submit(move || {
            for _ in 0..10 {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");
    assert_eq!(shared_counter.load(Ordering::SeqCst), 5000);
}

#[test]
fn test_rapid_task_submission() {
    let pool = ThreadPool::new().expect("Failed to create thread pool");
    let counter = Arc::new(AtomicU32::new(0));

    // Rapidly submit tasks without waiting
    for _ in 0..200 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            thread::sleep(Duration::from_micros(100));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");
    assert_eq!(counter.load(Ordering::SeqCst), 200);
}

#[test]
fn test_mixed_task_durations() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Mix of fast and slow tasks
    for i in 0..50 {
        let counter_clone = Arc::clone(&counter);
        let delay = if i % 5 == 0 {
            Duration::from_millis(20)
        } else {
            Duration::from_micros(100)
        };

        pool.submit(move || {
            thread::sleep(delay);
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");
    assert_eq!(counter.load(Ordering::SeqCst), 50);
}

#[test]
fn test_thread_pool_under_memory_pressure() {
    let pool = ThreadPool::new().expect("Failed to create thread pool");

    // Allocate memory in tasks
    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..100 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            let _vec: Vec<u8> = vec![0; 1024]; // 1KB allocation
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");
    assert_eq!(counter.load(Ordering::SeqCst), 100);
}

#[test]
fn test_work_stealing_pool_stress() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");
    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..500 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    // Wait for completion
    thread::sleep(Duration::from_secs(2));

    assert_eq!(counter.load(Ordering::SeqCst), 500);
}

#[test]
fn test_concurrent_pool_operations() {
    let pool = Arc::new(ThreadPool::new().expect("Failed to create thread pool"));
    let counter = Arc::new(AtomicU32::new(0));

    // Multiple threads submitting tasks concurrently
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let pool_clone = Arc::clone(&pool);
            let counter_clone = Arc::clone(&counter);

            thread::spawn(move || {
                for _ in 0..25 {
                    let counter_inner = Arc::clone(&counter_clone);
                    pool_clone
                        .submit(move || {
                            counter_inner.fetch_add(1, Ordering::SeqCst);
                        })
                        .expect("Failed to submit task");
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    pool.wait().expect("Failed to wait for tasks");
    assert_eq!(counter.load(Ordering::SeqCst), 100);
}

#[test]
fn test_queue_overflow_handling() {
    let config = ThreadPoolConfig {
        num_threads: Some(1),
        queue_capacity: 10,
        ..Default::default()
    };
    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    // Submit tasks slowly to avoid overwhelming the queue
    for _ in 0..10 {
        pool.submit(|| {
            thread::sleep(Duration::from_millis(10));
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");
}
