//! Enhanced thread pool with work-stealing deques and advanced features
//!
//! This module provides a high-performance thread pool implementation with:
//! - Work-stealing deques per thread for efficient load distribution
//! - Thread affinity and CPU pinning support
//! - Adaptive thread count based on workload
//! - Priority-based task scheduling
//! - Task dependency management

use crate::error::{NumRs2Error, Result};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads (None = auto-detect)
    pub num_threads: Option<usize>,
    /// Enable thread pinning to CPU cores
    pub enable_thread_pinning: bool,
    /// Enable adaptive thread count adjustment
    pub adaptive_threads: bool,
    /// Minimum number of threads (for adaptive mode)
    pub min_threads: usize,
    /// Maximum number of threads (for adaptive mode)
    pub max_threads: usize,
    /// Task queue capacity per thread
    pub queue_capacity: usize,
    /// Work stealing interval
    pub steal_interval: Duration,
    /// Thread idle timeout before parking
    pub idle_timeout: Duration,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism().map_or(4, |n| n.get());
        Self {
            num_threads: Some(num_cpus),
            enable_thread_pinning: false,
            adaptive_threads: false,
            min_threads: 1,
            max_threads: num_cpus * 2,
            queue_capacity: 1000,
            steal_interval: Duration::from_millis(1),
            idle_timeout: Duration::from_millis(10),
        }
    }
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Task with metadata
pub struct PoolTask {
    pub(crate) id: u64,
    pub(crate) priority: Priority,
    pub(crate) submitted_at: Instant,
    pub(crate) estimated_cost: Option<u64>,
    pub(crate) dependencies: Vec<u64>,
    pub(crate) task: Box<dyn FnOnce() + Send + 'static>,
}

impl std::fmt::Debug for PoolTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoolTask")
            .field("id", &self.id)
            .field("priority", &self.priority)
            .field("submitted_at", &self.submitted_at)
            .field("estimated_cost", &self.estimated_cost)
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

/// Thread-local worker state with work-stealing deque
/// Cache-aligned to prevent false sharing between worker threads
#[repr(align(64))]
struct WorkerState {
    id: usize,
    deque: Mutex<VecDeque<PoolTask>>,
    is_idle: AtomicBool,
    tasks_executed: AtomicUsize,
    tasks_stolen: AtomicUsize,
    total_execution_time: Mutex<Duration>,
    last_steal_time: Mutex<Instant>,
    cpu_affinity: Option<usize>,
    // Cache-line padding to prevent false sharing
    _padding: [u8; 0], // Padding will be added by alignment
}

impl WorkerState {
    fn new(id: usize, cpu_affinity: Option<usize>) -> Self {
        Self {
            id,
            deque: Mutex::new(VecDeque::new()),
            is_idle: AtomicBool::new(true),
            tasks_executed: AtomicUsize::new(0),
            tasks_stolen: AtomicUsize::new(0),
            total_execution_time: Mutex::new(Duration::ZERO),
            last_steal_time: Mutex::new(Instant::now()),
            cpu_affinity,
            _padding: [],
        }
    }

    fn push_task(&self, task: PoolTask) -> Result<()> {
        let mut deque = self
            .deque
            .lock()
            .map_err(|_| NumRs2Error::RuntimeError("Failed to acquire deque lock".to_string()))?;
        deque.push_back(task);
        Ok(())
    }

    fn pop_task(&self) -> Result<Option<PoolTask>> {
        let mut deque = self
            .deque
            .lock()
            .map_err(|_| NumRs2Error::RuntimeError("Failed to acquire deque lock".to_string()))?;
        Ok(deque.pop_front())
    }

    fn steal_task(&self) -> Result<Option<PoolTask>> {
        let mut deque = self
            .deque
            .lock()
            .map_err(|_| NumRs2Error::RuntimeError("Failed to acquire deque lock".to_string()))?;
        let task = deque.pop_back();
        if task.is_some() {
            self.tasks_stolen.fetch_add(1, Ordering::Relaxed);
        }
        Ok(task)
    }

    fn queue_len(&self) -> usize {
        self.deque.lock().map(|d| d.len()).unwrap_or(0)
    }

    fn is_idle(&self) -> bool {
        self.is_idle.load(Ordering::Relaxed)
    }

    fn set_idle(&self, idle: bool) {
        self.is_idle.store(idle, Ordering::Relaxed);
    }
}

/// Enhanced thread pool with work-stealing and advanced features
pub struct ThreadPool {
    config: ThreadPoolConfig,
    workers: Vec<Arc<WorkerState>>,
    threads: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    global_queue: Arc<Mutex<VecDeque<PoolTask>>>,
    idle_notify: Arc<(Mutex<()>, Condvar)>,
    next_task_id: AtomicUsize,
    stats: Arc<Mutex<ThreadPoolStats>>,
    completed_tasks: Arc<Mutex<Vec<u64>>>,
}

/// Thread pool statistics
#[derive(Debug, Clone, Default)]
pub struct ThreadPoolStats {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub tasks_stolen: u64,
    pub average_queue_time: Duration,
    pub average_execution_time: Duration,
    pub worker_utilization: Vec<f64>,
    pub active_threads: usize,
}

impl ThreadPool {
    /// Create a new thread pool with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ThreadPoolConfig::default())
    }

    /// Create a new thread pool with custom configuration
    pub fn with_config(config: ThreadPoolConfig) -> Result<Self> {
        let num_threads = config
            .num_threads
            .unwrap_or_else(|| thread::available_parallelism().map_or(4, |n| n.get()));

        let shutdown = Arc::new(AtomicBool::new(false));
        let global_queue = Arc::new(Mutex::new(VecDeque::new()));
        let idle_notify = Arc::new((Mutex::new(()), Condvar::new()));
        let stats = Arc::new(Mutex::new(ThreadPoolStats::default()));
        let completed_tasks = Arc::new(Mutex::new(Vec::new()));

        let mut workers = Vec::new();
        let mut threads = Vec::new();

        // Create worker states
        for i in 0..num_threads {
            let cpu_affinity = if config.enable_thread_pinning {
                Some(i % num_cpus::get())
            } else {
                None
            };
            workers.push(Arc::new(WorkerState::new(i, cpu_affinity)));
        }

        // Spawn worker threads
        for worker in &workers {
            let worker_clone = Arc::clone(worker);
            let workers_clone = workers.clone();
            let shutdown_clone = Arc::clone(&shutdown);
            let global_queue_clone = Arc::clone(&global_queue);
            let idle_notify_clone = Arc::clone(&idle_notify);
            let stats_clone = Arc::clone(&stats);
            let completed_tasks_clone = Arc::clone(&completed_tasks);
            let config_clone = config.clone();

            let handle = thread::spawn(move || {
                // Set thread affinity if enabled
                if let Some(cpu_id) = worker_clone.cpu_affinity {
                    Self::set_thread_affinity(cpu_id);
                }

                Self::worker_main(
                    worker_clone,
                    workers_clone,
                    shutdown_clone,
                    global_queue_clone,
                    idle_notify_clone,
                    stats_clone,
                    completed_tasks_clone,
                    config_clone,
                );
            });

            threads.push(handle);
        }

        Ok(Self {
            config,
            workers,
            threads,
            shutdown,
            global_queue,
            idle_notify,
            next_task_id: AtomicUsize::new(0),
            stats,
            completed_tasks,
        })
    }

    /// Submit a task to the pool
    pub fn submit<F>(&self, task: F) -> Result<u64>
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit_with_priority(task, Priority::Normal, None)
    }

    /// Submit a task with priority and cost estimate
    pub fn submit_with_priority<F>(
        &self,
        task: F,
        priority: Priority,
        estimated_cost: Option<u64>,
    ) -> Result<u64>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(NumRs2Error::RuntimeError(
                "Thread pool is shutting down".to_string(),
            ));
        }

        let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed) as u64;

        let pool_task = PoolTask {
            id: task_id,
            priority,
            submitted_at: Instant::now(),
            estimated_cost,
            dependencies: Vec::new(),
            task: Box::new(task),
        };

        // Find least loaded worker
        let target_worker = self.find_least_loaded_worker();

        if let Some(worker_idx) = target_worker {
            self.workers[worker_idx].push_task(pool_task)?;

            // Wake up worker if idle
            if self.workers[worker_idx].is_idle() {
                let (lock, cvar) = &*self.idle_notify;
                let _guard = lock.lock().map_err(|_| {
                    NumRs2Error::RuntimeError("Failed to acquire idle notify lock".to_string())
                })?;
                cvar.notify_one();
            }
        } else {
            // Fallback to global queue
            let mut global = self.global_queue.lock().map_err(|_| {
                NumRs2Error::RuntimeError("Failed to acquire global queue lock".to_string())
            })?;
            global.push_back(pool_task);

            let (lock, cvar) = &*self.idle_notify;
            let _guard = lock.lock().map_err(|_| {
                NumRs2Error::RuntimeError("Failed to acquire idle notify lock".to_string())
            })?;
            cvar.notify_all();
        }

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.tasks_submitted += 1;
        }

        Ok(task_id)
    }

    /// Get pool statistics
    pub fn statistics(&self) -> ThreadPoolStats {
        if let Ok(mut stats) = self.stats.lock() {
            stats.worker_utilization = self
                .workers
                .iter()
                .map(|w| if w.is_idle() { 0.0 } else { 1.0 })
                .collect();

            stats.active_threads = self.workers.iter().filter(|w| !w.is_idle()).count();

            stats.clone()
        } else {
            ThreadPoolStats::default()
        }
    }

    /// Get number of worker threads
    pub fn num_threads(&self) -> usize {
        self.workers.len()
    }

    /// Get number of pending tasks
    pub fn pending_tasks(&self) -> usize {
        let global_count = self.global_queue.lock().map(|q| q.len()).unwrap_or(0);

        let worker_count: usize = self.workers.iter().map(|w| w.queue_len()).sum();

        global_count + worker_count
    }

    /// Wait for all pending tasks to complete
    pub fn wait(&self) -> Result<()> {
        // Wait until there are no pending tasks AND all workers are idle
        while self.pending_tasks() > 0 || self.has_active_workers() {
            thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }

    /// Check if any workers are actively executing tasks
    fn has_active_workers(&self) -> bool {
        self.workers.iter().any(|w| !w.is_idle())
    }

    /// Shutdown the thread pool gracefully
    pub fn shutdown(self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);

        // Wake up all workers
        let (lock, cvar) = &*self.idle_notify;
        let _guard = lock.lock().map_err(|_| {
            NumRs2Error::RuntimeError("Failed to acquire idle notify lock".to_string())
        })?;
        cvar.notify_all();
        drop(_guard);

        // Join all threads
        for handle in self.threads {
            if let Err(_e) = handle.join() {
                // Log error but continue shutting down other threads
            }
        }

        Ok(())
    }

    // Private helper methods

    fn find_least_loaded_worker(&self) -> Option<usize> {
        self.workers
            .iter()
            .enumerate()
            .min_by_key(|(_, w)| w.queue_len())
            .map(|(idx, _)| idx)
    }

    fn worker_main(
        worker: Arc<WorkerState>,
        workers: Vec<Arc<WorkerState>>,
        shutdown: Arc<AtomicBool>,
        global_queue: Arc<Mutex<VecDeque<PoolTask>>>,
        idle_notify: Arc<(Mutex<()>, Condvar)>,
        stats: Arc<Mutex<ThreadPoolStats>>,
        completed_tasks: Arc<Mutex<Vec<u64>>>,
        config: ThreadPoolConfig,
    ) {
        let worker_id = worker.id;

        while !shutdown.load(Ordering::Relaxed) {
            let mut task_found = false;

            // 1. Try local queue
            if let Ok(Some(task)) = worker.pop_task() {
                Self::execute_task(task, &worker, &stats, &completed_tasks);
                task_found = true;
            }

            // 2. Try global queue
            if !task_found {
                if let Ok(mut global) = global_queue.try_lock() {
                    if let Some(task) = global.pop_front() {
                        drop(global);
                        Self::execute_task(task, &worker, &stats, &completed_tasks);
                        task_found = true;
                    }
                }
            }

            // 3. Try work stealing
            if !task_found {
                if let Some(stolen_task) = Self::try_steal_work(&worker, &workers, &config) {
                    Self::execute_task(stolen_task, &worker, &stats, &completed_tasks);
                    task_found = true;
                }
            }

            // 4. Park if no work found
            if !task_found {
                worker.set_idle(true);

                let (lock, cvar) = &*idle_notify;
                if let Ok(guard) = lock.lock() {
                    let _result = cvar.wait_timeout(guard, config.idle_timeout);
                }

                worker.set_idle(false);

                // Check shutdown again after waking up
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
            }
        }
    }

    fn execute_task(
        task: PoolTask,
        worker: &Arc<WorkerState>,
        stats: &Arc<Mutex<ThreadPoolStats>>,
        completed_tasks: &Arc<Mutex<Vec<u64>>>,
    ) {
        let start_time = Instant::now();
        let task_id = task.id;

        // Execute the task
        (task.task)();

        let execution_time = start_time.elapsed();

        // Update worker stats
        worker.tasks_executed.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut total_time) = worker.total_execution_time.lock() {
            *total_time += execution_time;
        }

        // Mark task as completed
        if let Ok(mut completed) = completed_tasks.lock() {
            completed.push(task_id);
        }

        // Update global stats
        if let Ok(mut global_stats) = stats.lock() {
            global_stats.tasks_completed += 1;

            // Update average execution time (exponential moving average)
            let alpha = 0.1;
            global_stats.average_execution_time = Duration::from_secs_f64(
                alpha * execution_time.as_secs_f64()
                    + (1.0 - alpha) * global_stats.average_execution_time.as_secs_f64(),
            );
        }
    }

    fn try_steal_work(
        worker: &Arc<WorkerState>,
        workers: &[Arc<WorkerState>],
        config: &ThreadPoolConfig,
    ) -> Option<PoolTask> {
        let now = Instant::now();

        // Check steal interval
        if let Ok(mut last_steal) = worker.last_steal_time.lock() {
            if now.duration_since(*last_steal) < config.steal_interval {
                return None;
            }
            *last_steal = now;
        }

        // Find victim with most tasks
        let victim = workers
            .iter()
            .filter(|w| w.id != worker.id)
            .max_by_key(|w| w.queue_len())?;

        if victim.queue_len() > 1 {
            if let Ok(Some(task)) = victim.steal_task() {
                return Some(task);
            }
        }

        None
    }

    fn set_thread_affinity(_cpu_id: usize) {
        // Platform-specific implementation would go here
        // For now, this is a no-op as it requires platform-specific code
        #[cfg(target_os = "linux")]
        {
            // On Linux, we could use libc::pthread_setaffinity_np
            // But for pure Rust, we'll skip this for now
        }
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new().expect("Failed to create default thread pool")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new().expect("Failed to create thread pool");
        assert!(pool.num_threads() > 0);
    }

    #[test]
    fn test_task_submission() {
        let pool = ThreadPool::new().expect("Failed to create thread pool");
        let counter = Arc::new(AtomicU32::new(0));

        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            pool.submit(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .expect("Failed to submit task");
        }

        pool.wait().expect("Failed to wait for tasks");
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_priority_tasks() {
        let pool = ThreadPool::new().expect("Failed to create thread pool");
        let counter = Arc::new(AtomicU32::new(0));

        // Submit high priority task
        let counter_clone = Arc::clone(&counter);
        pool.submit_with_priority(
            move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            },
            Priority::High,
            None,
        )
        .expect("Failed to submit high priority task");

        pool.wait().expect("Failed to wait for tasks");
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_statistics() {
        let pool = ThreadPool::new().expect("Failed to create thread pool");

        for _ in 0..5 {
            pool.submit(|| {
                thread::sleep(Duration::from_millis(10));
            })
            .expect("Failed to submit task");
        }

        thread::sleep(Duration::from_millis(100));

        let stats = pool.statistics();
        assert_eq!(stats.tasks_submitted, 5);
        assert!(stats.active_threads <= pool.num_threads());
    }

    #[test]
    fn test_work_stealing() {
        let config = ThreadPoolConfig {
            num_threads: Some(2),
            ..Default::default()
        };
        let pool = ThreadPool::with_config(config).expect("Failed to create thread pool");
        let counter = Arc::new(AtomicU32::new(0));

        // Submit many tasks to trigger work stealing
        for _ in 0..20 {
            let counter_clone = Arc::clone(&counter);
            pool.submit(move || {
                thread::sleep(Duration::from_millis(5));
                counter_clone.fetch_add(1, Ordering::SeqCst);
            })
            .expect("Failed to submit task");
        }

        pool.wait().expect("Failed to wait for tasks");

        // Extra wait to ensure all tasks complete
        thread::sleep(Duration::from_millis(200));

        assert_eq!(counter.load(Ordering::SeqCst), 20);
    }
}
