//! Tests for work-stealing correctness

use numrs2::parallel::{ThreadPool, ThreadPoolConfig, WorkStealingPool, task};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_work_stealing_basic() {
    let pool = ThreadPool::with_config(ThreadPoolConfig {
        num_threads: Some(4),
        ..Default::default()
    })
    .expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // Submit many tasks to trigger work stealing
    for _ in 0..100 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");
    assert_eq!(counter.load(Ordering::SeqCst), 100);
}

#[test]
fn test_work_stealing_imbalanced_load() {
    let config = ThreadPoolConfig {
        num_threads: Some(2),
        ..Default::default()
    };
    let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");

    let counter = Arc::new(AtomicU32::new(0));

    // First thread gets many long tasks
    for _ in 0..5 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            thread::sleep(Duration::from_millis(50));
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    // Give time for tasks to start on one thread
    thread::sleep(Duration::from_millis(10));

    // Submit more tasks that should be stolen
    for _ in 0..10 {
        let counter_clone = Arc::clone(&counter);
        pool.submit(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");
    assert_eq!(counter.load(Ordering::SeqCst), 15);

    let stats = pool.statistics();
    assert_eq!(stats.tasks_submitted, 15);
}

#[test]
fn test_work_stealing_pool_correctness() {
    let pool = WorkStealingPool::new(4).expect("Failed to create work-stealing pool");

    let counter = Arc::new(AtomicU32::new(0));

    for _ in 0..50 {
        let counter_clone = Arc::clone(&counter);
        let task = task(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        pool.submit(task).expect("Failed to submit task");
    }

    // Wait for completion
    thread::sleep(Duration::from_millis(500));

    assert_eq!(counter.load(Ordering::SeqCst), 50);
}

#[test]
fn test_work_stealing_no_data_races() {
    let pool = ThreadPool::new().expect("Failed to create thread pool");

    let shared_vec = Arc::new(std::sync::Mutex::new(Vec::new()));

    for i in 0..100 {
        let vec_clone = Arc::clone(&shared_vec);
        pool.submit(move || {
            let mut vec = vec_clone.lock().expect("Failed to lock shared vec");
            vec.push(i);
        })
        .expect("Failed to submit task");
    }

    pool.wait().expect("Failed to wait for tasks");

    let vec = shared_vec.lock().expect("Failed to lock shared vec");
    assert_eq!(vec.len(), 100);
}

#[test]
fn test_work_stealing_statistics() {
    let pool = ThreadPool::new().expect("Failed to create thread pool");

    for _ in 0..20 {
        pool.submit(|| {
            thread::sleep(Duration::from_millis(10));
        })
        .expect("Failed to submit task");
    }

    thread::sleep(Duration::from_millis(300));

    let stats = pool.statistics();
    assert_eq!(stats.tasks_submitted, 20);
    assert!(stats.worker_utilization.len() > 0);
}
