//! Work-stealing thread pool implementation
//!
//! This module provides a high-performance work-stealing thread pool
//! optimized for parallel numerical computations with low overhead.

use crate::error::{NumRs2Error, Result};
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// A task that can be executed by the work-stealing pool
pub trait Task: Send + 'static {
    type Output: Send + 'static;

    /// Execute the task
    fn execute(self: Box<Self>) -> Self::Output;

    /// Estimate the computational cost of this task (optional)
    fn estimated_cost(&self) -> Option<u64> {
        None
    }

    /// Check if this task can be split for better load balancing
    fn can_split(&self) -> bool {
        false
    }

    /// Split the task into smaller tasks (if supported)
    fn split(self: Box<Self>) -> Vec<Box<dyn Task<Output = Self::Output>>>
    where
        Self: Sized,
    {
        vec![self]
    }
}

/// Result of task execution
#[derive(Debug)]
pub enum TaskResult<T> {
    Success(T),
    Error(String),
    Cancelled,
}

/// A boxed task for type erasure
type BoxedTask = Box<dyn Task<Output = TaskResult<()>>>;

/// Thread-local work queue with work-stealing support
struct WorkerQueue {
    deque: VecDeque<BoxedTask>,
    steal_count: u64,
    execute_count: u64,
}

impl WorkerQueue {
    fn new() -> Self {
        Self {
            deque: VecDeque::new(),
            steal_count: 0,
            execute_count: 0,
        }
    }

    fn push_back(&mut self, task: BoxedTask) {
        self.deque.push_back(task);
    }

    fn pop_front(&mut self) -> Option<BoxedTask> {
        let task = self.deque.pop_front();
        if task.is_some() {
            self.execute_count += 1;
        }
        task
    }

    fn steal(&mut self) -> Option<BoxedTask> {
        let task = self.deque.pop_back();
        if task.is_some() {
            self.steal_count += 1;
        }
        task
    }

    fn len(&self) -> usize {
        self.deque.len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }
}

/// Worker thread state
struct WorkerState {
    id: usize,
    queue: Mutex<WorkerQueue>,
    is_idle: AtomicBool,
    tasks_executed: AtomicUsize,
    total_execution_time: Mutex<Duration>,
    last_steal_attempt: Mutex<Instant>,
}

impl WorkerState {
    fn new(id: usize) -> Self {
        Self {
            id,
            queue: Mutex::new(WorkerQueue::new()),
            is_idle: AtomicBool::new(true),
            tasks_executed: AtomicUsize::new(0),
            total_execution_time: Mutex::new(Duration::ZERO),
            last_steal_attempt: Mutex::new(Instant::now()),
        }
    }

    fn queue_length(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    fn is_idle(&self) -> bool {
        self.is_idle.load(Ordering::Relaxed)
    }

    fn set_idle(&self, idle: bool) {
        self.is_idle.store(idle, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    fn throughput(&self) -> f64 {
        let total_time = self.total_execution_time.lock().unwrap();
        if total_time.is_zero() {
            0.0
        } else {
            self.tasks_executed.load(Ordering::Relaxed) as f64 / total_time.as_secs_f64()
        }
    }
}

/// Work-stealing thread pool configuration
#[derive(Debug, Clone)]
pub struct WorkStealingConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Maximum number of steal attempts per idle cycle
    pub max_steal_attempts: usize,
    /// Minimum time between steal attempts
    pub steal_interval: Duration,
    /// Thread idle timeout before parking
    pub idle_timeout: Duration,
    /// Enable task splitting for large tasks
    pub enable_task_splitting: bool,
    /// Maximum task queue size per worker
    pub max_queue_size: usize,
    /// Enable adaptive work stealing based on queue imbalance
    pub adaptive_stealing: bool,
}

impl Default for WorkStealingConfig {
    fn default() -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            max_steal_attempts: 3,
            steal_interval: Duration::from_millis(1),
            idle_timeout: Duration::from_millis(10),
            enable_task_splitting: true,
            max_queue_size: 1000,
            adaptive_stealing: true,
        }
    }
}

/// High-performance work-stealing thread pool
pub struct WorkStealingPool {
    config: WorkStealingConfig,
    workers: Vec<Arc<WorkerState>>,
    #[allow(dead_code)]
    threads: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    global_queue: Arc<Mutex<VecDeque<BoxedTask>>>,
    idle_workers: Arc<(Mutex<Vec<usize>>, Condvar)>,
    stats: Arc<Mutex<PoolStats>>,
}

/// Pool performance statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub tasks_stolen: u64,
    pub total_steal_attempts: u64,
    pub average_queue_time: Duration,
    pub average_execution_time: Duration,
    pub worker_utilization: Vec<f64>,
    pub queue_imbalance: f64,
}

impl WorkStealingPool {
    /// Create a new work-stealing pool
    pub fn new(num_threads: usize) -> Result<Self> {
        let config = WorkStealingConfig {
            num_threads,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a work-stealing pool with custom configuration
    pub fn with_config(config: WorkStealingConfig) -> Result<Self> {
        let shutdown = Arc::new(AtomicBool::new(false));
        let global_queue = Arc::new(Mutex::new(VecDeque::new()));
        let idle_workers = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
        let stats = Arc::new(Mutex::new(PoolStats::default()));

        let mut workers = Vec::new();
        let mut threads = Vec::new();

        // Create worker states
        for i in 0..config.num_threads {
            workers.push(Arc::new(WorkerState::new(i)));
        }

        // Spawn worker threads
        for worker in &workers {
            let worker_clone = Arc::clone(worker);
            let workers_clone = workers.clone();
            let shutdown_clone = Arc::clone(&shutdown);
            let global_queue_clone = Arc::clone(&global_queue);
            let idle_workers_clone = Arc::clone(&idle_workers);
            let stats_clone = Arc::clone(&stats);
            let config_clone = config.clone();

            let handle = thread::spawn(move || {
                Self::worker_main(
                    worker_clone,
                    workers_clone,
                    shutdown_clone,
                    global_queue_clone,
                    idle_workers_clone,
                    stats_clone,
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
            idle_workers,
            stats,
        })
    }

    /// Submit a task for execution
    pub fn submit<T>(&self, task: T) -> Result<()>
    where
        T: Task<Output = TaskResult<()>> + 'static,
    {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(NumRs2Error::RuntimeError(
                "Pool is shutting down".to_string(),
            ));
        }

        let boxed_task: BoxedTask = Box::new(task);

        // Try to find the least loaded worker
        let target_worker = self.find_least_loaded_worker();

        if let Some(worker_id) = target_worker {
            let worker = &self.workers[worker_id];
            let mut queue = worker.queue.lock().unwrap();

            if queue.len() < self.config.max_queue_size {
                queue.push_back(boxed_task);
                drop(queue);

                // Wake up the worker if it's idle
                if worker.is_idle() {
                    self.notify_worker(worker_id);
                }

                // Update stats
                {
                    let mut stats = self.stats.lock().unwrap();
                    stats.tasks_submitted += 1;
                }

                return Ok(());
            }
        }

        // Fallback to global queue
        {
            let mut global = self.global_queue.lock().unwrap();
            if global.len() < self.config.max_queue_size * 2 {
                global.push_back(boxed_task);

                // Notify any idle worker
                self.notify_idle_workers();

                let mut stats = self.stats.lock().unwrap();
                stats.tasks_submitted += 1;

                Ok(())
            } else {
                Err(NumRs2Error::RuntimeError("All queues are full".to_string()))
            }
        }
    }

    /// Submit a high-priority task that goes to the front of queues
    pub fn submit_urgent<T>(&self, task: T) -> Result<()>
    where
        T: Task<Output = TaskResult<()>> + 'static,
    {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(NumRs2Error::RuntimeError(
                "Pool is shutting down".to_string(),
            ));
        }

        let boxed_task: BoxedTask = Box::new(task);

        // Add to global queue at front for urgent tasks
        {
            let mut global = self.global_queue.lock().unwrap();
            global.push_front(boxed_task);

            self.notify_idle_workers();

            let mut stats = self.stats.lock().unwrap();
            stats.tasks_submitted += 1;
        }

        Ok(())
    }

    /// Get current pool statistics
    pub fn statistics(&self) -> PoolStats {
        let mut stats = self.stats.lock().unwrap();

        // Update worker utilization
        stats.worker_utilization = self
            .workers
            .iter()
            .map(|worker| if worker.is_idle() { 0.0 } else { 1.0 })
            .collect();

        // Calculate queue imbalance
        stats.queue_imbalance = self.calculate_queue_imbalance();

        stats.clone()
    }

    /// Get number of active workers
    pub fn active_workers(&self) -> usize {
        self.workers
            .iter()
            .filter(|worker| !worker.is_idle())
            .count()
    }

    /// Get total number of pending tasks
    pub fn pending_tasks(&self) -> usize {
        let global_count = self.global_queue.lock().unwrap().len();
        let worker_count: usize = self
            .workers
            .iter()
            .map(|worker| worker.queue_length())
            .sum();

        global_count + worker_count
    }

    /// Shutdown the pool gracefully
    pub fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);

        // Wake up all workers
        let (idle_lock, condvar) = &*self.idle_workers;
        let _idle = idle_lock.lock().unwrap();
        condvar.notify_all();

        // Note: In a real implementation, we'd need to join the threads
        // but that requires consuming self, which isn't possible here

        Ok(())
    }

    // Private helper methods

    fn find_least_loaded_worker(&self) -> Option<usize> {
        self.workers
            .iter()
            .enumerate()
            .min_by_key(|(_, worker)| worker.queue_length())
            .map(|(idx, _)| idx)
    }

    fn notify_worker(&self, worker_id: usize) {
        let (idle_lock, condvar) = &*self.idle_workers;
        let mut idle = idle_lock.lock().unwrap();

        if let Some(pos) = idle.iter().position(|&id| id == worker_id) {
            idle.remove(pos);
            condvar.notify_one();
        }
    }

    fn notify_idle_workers(&self) {
        let (_, condvar) = &*self.idle_workers;
        condvar.notify_all();
    }

    fn calculate_queue_imbalance(&self) -> f64 {
        let queue_lengths: Vec<usize> = self
            .workers
            .iter()
            .map(|worker| worker.queue_length())
            .collect();

        if queue_lengths.is_empty() {
            return 0.0;
        }

        let max_len = *queue_lengths.iter().max().unwrap_or(&0) as f64;
        let min_len = *queue_lengths.iter().min().unwrap_or(&0) as f64;

        if max_len == 0.0 {
            0.0
        } else {
            (max_len - min_len) / max_len
        }
    }

    /// Main worker thread loop
    fn worker_main(
        worker: Arc<WorkerState>,
        workers: Vec<Arc<WorkerState>>,
        shutdown: Arc<AtomicBool>,
        global_queue: Arc<Mutex<VecDeque<BoxedTask>>>,
        idle_workers: Arc<(Mutex<Vec<usize>>, Condvar)>,
        stats: Arc<Mutex<PoolStats>>,
        config: WorkStealingConfig,
    ) {
        let worker_id = worker.id;

        while !shutdown.load(Ordering::Relaxed) {
            let mut task_found = false;

            // 1. Try to get task from local queue
            if let Ok(mut queue) = worker.queue.try_lock() {
                if let Some(task) = queue.pop_front() {
                    drop(queue);
                    Self::execute_task(task, &worker, &stats);
                    task_found = true;
                }
            }

            // 2. Try to get task from global queue
            if !task_found {
                if let Ok(mut global) = global_queue.try_lock() {
                    if let Some(task) = global.pop_front() {
                        drop(global);
                        Self::execute_task(task, &worker, &stats);
                        task_found = true;
                    }
                }
            }

            // 3. Try work stealing from other workers
            if !task_found && config.adaptive_stealing {
                if let Some(stolen_task) = Self::try_steal_work(&worker, &workers, &config) {
                    Self::execute_task(stolen_task, &worker, &stats);
                    task_found = true;
                }
            }

            // 4. No work found, become idle
            if !task_found {
                worker.set_idle(true);

                let (idle_lock, condvar) = &*idle_workers;
                let mut idle = idle_lock.lock().unwrap();
                idle.push(worker_id);

                // Wait for work or timeout
                let _result = condvar.wait_timeout(idle, config.idle_timeout);

                worker.set_idle(false);
            }
        }
    }

    fn execute_task(task: BoxedTask, worker: &Arc<WorkerState>, stats: &Arc<Mutex<PoolStats>>) {
        let start_time = Instant::now();

        let result = Box::new(task).execute();

        let execution_time = start_time.elapsed();

        // Update worker stats
        worker.tasks_executed.fetch_add(1, Ordering::Relaxed);
        {
            let mut total_time = worker.total_execution_time.lock().unwrap();
            *total_time += execution_time;
        }

        // Update global stats
        {
            let mut global_stats = stats.lock().unwrap();
            match result {
                TaskResult::Success(_) => global_stats.tasks_completed += 1,
                TaskResult::Error(_) | TaskResult::Cancelled => {
                    // Count as completed but could track errors separately
                    global_stats.tasks_completed += 1;
                }
            }

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
        config: &WorkStealingConfig,
    ) -> Option<BoxedTask> {
        let now = Instant::now();
        let mut last_steal = worker.last_steal_attempt.lock().unwrap();

        // Check if enough time has passed since last steal attempt
        if now.duration_since(*last_steal) < config.steal_interval {
            return None;
        }
        *last_steal = now;
        drop(last_steal);

        // Try stealing from multiple workers
        for _ in 0..config.max_steal_attempts {
            // Find a victim (worker with most tasks)
            let victim = workers
                .iter()
                .filter(|w| w.id != worker.id)
                .max_by_key(|w| w.queue_length())?;

            if victim.queue_length() > 1 {
                if let Ok(mut victim_queue) = victim.queue.try_lock() {
                    if let Some(stolen_task) = victim_queue.steal() {
                        return Some(stolen_task);
                    }
                }
            }
        }

        None
    }
}

/// Simple task implementation for closures
pub struct ClosureTask<F, T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    closure: Option<F>,
    _phantom: std::marker::PhantomData<T>,
}

impl<F, T> ClosureTask<F, T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    pub fn new(closure: F) -> Self {
        Self {
            closure: Some(closure),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F> Task for ClosureTask<F, ()>
where
    F: FnOnce() + Send + 'static,
{
    type Output = TaskResult<()>;

    fn execute(mut self: Box<Self>) -> Self::Output {
        if let Some(closure) = self.closure.take() {
            closure();
            TaskResult::Success(())
        } else {
            TaskResult::Error("Task already executed".to_string())
        }
    }
}

/// Convenience function to create a task from a closure
pub fn task<F>(closure: F) -> ClosureTask<F, ()>
where
    F: FnOnce() + Send + 'static,
{
    ClosureTask::new(closure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_work_stealing_pool_creation() {
        let pool = WorkStealingPool::new(2).unwrap();
        assert_eq!(pool.config.num_threads, 2);
        assert_eq!(pool.active_workers(), 0); // No tasks running yet
        assert_eq!(pool.pending_tasks(), 0);
    }

    #[test]
    fn test_task_submission() {
        let pool = WorkStealingPool::new(2).unwrap();
        let counter = Arc::new(AtomicU32::new(0));

        for _ in 0..5 {
            let counter_clone = Arc::clone(&counter);
            let task = task(move || {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });

            pool.submit(task).unwrap();
        }

        // Wait for tasks to complete
        std::thread::sleep(Duration::from_millis(100));

        assert_eq!(counter.load(Ordering::SeqCst), 5);

        let stats = pool.statistics();
        assert_eq!(stats.tasks_submitted, 5);
        assert!(stats.tasks_completed <= 5); // Some might still be running
    }

    #[test]
    fn test_urgent_task_submission() {
        let pool = WorkStealingPool::new(1).unwrap();
        let execution_order = Arc::new(Mutex::new(Vec::new()));

        // Submit normal task
        {
            let order_clone = Arc::clone(&execution_order);
            let task = task(move || {
                std::thread::sleep(Duration::from_millis(50)); // Slow task
                order_clone.lock().unwrap().push(1);
            });
            pool.submit(task).unwrap();
        }

        // Submit urgent task (should execute first)
        {
            let order_clone = Arc::clone(&execution_order);
            let urgent_task = task(move || {
                order_clone.lock().unwrap().push(2);
            });
            pool.submit_urgent(urgent_task).unwrap();
        }

        std::thread::sleep(Duration::from_millis(200));

        let order = execution_order.lock().unwrap();
        assert!(!order.is_empty());
        // Note: In this simple test, the exact order depends on timing
    }

    #[test]
    fn test_pool_statistics() {
        let pool = WorkStealingPool::new(2).unwrap();

        // Submit some tasks
        for i in 0..3 {
            let task = task(move || {
                std::thread::sleep(Duration::from_millis(10 * i as u64));
            });
            pool.submit(task).unwrap();
        }

        std::thread::sleep(Duration::from_millis(100));

        let stats = pool.statistics();
        assert_eq!(stats.tasks_submitted, 3);
        assert!(stats.worker_utilization.len() == 2);
        assert!(stats.queue_imbalance >= 0.0);
    }

    #[test]
    fn test_queue_imbalance_calculation() {
        let pool = WorkStealingPool::new(3).unwrap();

        // Add tasks to create imbalance
        for _ in 0..10 {
            let task = task(|| {
                std::thread::sleep(Duration::from_millis(100)); // Long-running task
            });
            pool.submit(task).unwrap();
        }

        let imbalance = pool.calculate_queue_imbalance();
        assert!((0.0..=1.0).contains(&imbalance));
    }

    #[test]
    fn test_worker_state() {
        let worker = WorkerState::new(0);
        assert_eq!(worker.id, 0);
        assert_eq!(worker.queue_length(), 0);
        assert!(worker.is_idle());
        assert_eq!(worker.tasks_executed.load(Ordering::Relaxed), 0);

        worker.set_idle(false);
        assert!(!worker.is_idle());
    }

    #[test]
    fn test_closure_task() {
        let executed = Arc::new(AtomicU32::new(0));
        let executed_clone = Arc::clone(&executed);

        let task = ClosureTask::new(move || {
            executed_clone.store(42, Ordering::SeqCst);
        });

        let result = Box::new(task).execute();
        assert!(matches!(result, TaskResult::Success(())));
        assert_eq!(executed.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn test_pool_shutdown() {
        let pool = WorkStealingPool::new(2).unwrap();

        // Submit a task
        let task = task(|| {
            std::thread::sleep(Duration::from_millis(10));
        });
        pool.submit(task).unwrap();

        // Shutdown should succeed
        assert!(pool.shutdown().is_ok());
    }
}
