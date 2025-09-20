//! Advanced parallel task scheduler with priority management
//!
//! This module provides a sophisticated task scheduler that manages parallel
//! execution with priority queues, thread affinity, and adaptive scheduling.

use crate::error::{NumRs2Error, Result};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Task priority levels for scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Configuration for the parallel scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of worker threads
    pub num_threads: usize,
    /// Maximum queue size per priority level
    pub max_queue_size: usize,
    /// Thread affinity settings
    pub enable_thread_affinity: bool,
    /// Adaptive scheduling parameters
    pub enable_adaptive_scheduling: bool,
    /// Time slice for time-sharing
    pub time_slice_ms: u64,
    /// Work stealing threshold
    pub work_stealing_threshold: usize,
    /// CPU cache awareness
    pub cache_aware_scheduling: bool,
}

impl SchedulerConfig {
    /// Create optimal configuration for the given number of cores
    pub fn optimal_for_cores(num_cores: usize) -> Self {
        Self {
            num_threads: num_cores,
            max_queue_size: 1000,
            enable_thread_affinity: true,
            enable_adaptive_scheduling: true,
            time_slice_ms: 10,
            work_stealing_threshold: 5,
            cache_aware_scheduling: true,
        }
    }

    /// Create configuration optimized for throughput
    pub fn throughput_optimized(num_cores: usize) -> Self {
        Self {
            num_threads: num_cores,
            max_queue_size: 2000,
            enable_thread_affinity: false,
            enable_adaptive_scheduling: true,
            time_slice_ms: 5,
            work_stealing_threshold: 3,
            cache_aware_scheduling: false,
        }
    }

    /// Create configuration optimized for latency
    pub fn latency_optimized(num_cores: usize) -> Self {
        Self {
            num_threads: num_cores,
            max_queue_size: 500,
            enable_thread_affinity: true,
            enable_adaptive_scheduling: false,
            time_slice_ms: 2,
            work_stealing_threshold: 8,
            cache_aware_scheduling: true,
        }
    }
}

/// A scheduled task with priority and execution context
pub struct ScheduledTask {
    pub id: u64,
    pub priority: TaskPriority,
    pub submitted_at: Instant,
    pub estimated_duration: Option<Duration>,
    pub thread_affinity: Option<usize>,
    pub task: Box<dyn FnOnce() -> TaskResult + Send + 'static>,
}

impl std::fmt::Debug for ScheduledTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScheduledTask")
            .field("id", &self.id)
            .field("priority", &self.priority)
            .field("submitted_at", &self.submitted_at)
            .field("estimated_duration", &self.estimated_duration)
            .field("thread_affinity", &self.thread_affinity)
            .field("task", &"<closure>")
            .finish()
    }
}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.submitted_at == other.submitted_at
    }
}

impl Eq for ScheduledTask {}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier submission time
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.submitted_at.cmp(&self.submitted_at), // Earlier tasks first
            other => other,                                                // Higher priority first
        }
    }
}

/// Result of task execution
#[derive(Debug)]
pub enum TaskResult {
    Success,
    Error(String),
    Cancelled,
}

/// Thread-local scheduler state
#[derive(Debug)]
struct ThreadState {
    #[allow(dead_code)]
    id: usize,
    local_queue: BinaryHeap<ScheduledTask>,
    tasks_executed: u64,
    total_execution_time: Duration,
    idle_time: Duration,
    last_steal_time: Instant,
}

impl ThreadState {
    fn new(id: usize) -> Self {
        Self {
            id,
            local_queue: BinaryHeap::new(),
            tasks_executed: 0,
            total_execution_time: Duration::ZERO,
            idle_time: Duration::ZERO,
            last_steal_time: Instant::now(),
        }
    }

    fn efficiency(&self) -> f64 {
        let total_time = self.total_execution_time + self.idle_time;
        if total_time.is_zero() {
            0.0
        } else {
            self.total_execution_time.as_secs_f64() / total_time.as_secs_f64()
        }
    }
}

/// Advanced parallel scheduler with priority management
pub struct ParallelScheduler {
    config: SchedulerConfig,
    global_queue: Arc<Mutex<BinaryHeap<ScheduledTask>>>,
    #[allow(dead_code)]
    worker_threads: Vec<JoinHandle<()>>,
    shutdown_signal: Arc<(Mutex<bool>, Condvar)>,
    thread_states: Arc<Mutex<Vec<ThreadState>>>,
    next_task_id: Arc<Mutex<u64>>,
    scheduler_stats: Arc<Mutex<SchedulerStats>>,
}

/// Scheduler performance statistics
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub average_queue_time: Duration,
    pub average_execution_time: Duration,
    pub thread_efficiency: Vec<f64>,
    pub work_steals: u64,
    pub queue_overflows: u64,
}

impl ParallelScheduler {
    /// Create a new parallel scheduler
    pub fn new(config: SchedulerConfig) -> Result<Self> {
        let global_queue = Arc::new(Mutex::new(BinaryHeap::new()));
        let shutdown_signal = Arc::new((Mutex::new(false), Condvar::new()));
        let thread_states = Arc::new(Mutex::new(Vec::new()));
        let next_task_id = Arc::new(Mutex::new(0));
        let scheduler_stats = Arc::new(Mutex::new(SchedulerStats::default()));

        // Initialize thread states
        {
            let mut states = thread_states.lock().unwrap();
            for i in 0..config.num_threads {
                states.push(ThreadState::new(i));
            }
        }

        let mut worker_threads = Vec::new();

        // Spawn worker threads
        for thread_id in 0..config.num_threads {
            let global_queue = Arc::clone(&global_queue);
            let shutdown_signal = Arc::clone(&shutdown_signal);
            let thread_states = Arc::clone(&thread_states);
            let scheduler_stats = Arc::clone(&scheduler_stats);
            let config = config.clone();

            let handle = thread::spawn(move || {
                Self::worker_thread_main(
                    thread_id,
                    global_queue,
                    shutdown_signal,
                    thread_states,
                    scheduler_stats,
                    config,
                );
            });

            worker_threads.push(handle);
        }

        Ok(Self {
            config,
            global_queue,
            worker_threads,
            shutdown_signal,
            thread_states,
            next_task_id,
            scheduler_stats,
        })
    }

    /// Submit a task for execution
    pub fn submit_task<F>(
        &self,
        task: F,
        priority: TaskPriority,
        estimated_duration: Option<Duration>,
        thread_affinity: Option<usize>,
    ) -> Result<u64>
    where
        F: FnOnce() -> TaskResult + Send + 'static,
    {
        let task_id = {
            let mut id = self.next_task_id.lock().unwrap();
            *id += 1;
            *id
        };

        let scheduled_task = ScheduledTask {
            id: task_id,
            priority,
            submitted_at: Instant::now(),
            estimated_duration,
            thread_affinity,
            task: Box::new(task),
        };

        let mut queue = self.global_queue.lock().unwrap();

        // Check queue capacity
        if queue.len() >= self.config.max_queue_size {
            let mut stats = self.scheduler_stats.lock().unwrap();
            stats.queue_overflows += 1;
            drop(stats);
            drop(queue);
            return Err(NumRs2Error::RuntimeError("Task queue full".to_string()));
        }

        queue.push(scheduled_task);

        {
            let mut stats = self.scheduler_stats.lock().unwrap();
            stats.tasks_submitted += 1;
        }

        // Notify waiting workers
        let (_, condvar) = &*self.shutdown_signal;
        condvar.notify_one();

        Ok(task_id)
    }

    /// Submit a high-priority task
    pub fn submit_urgent_task<F>(&self, task: F) -> Result<u64>
    where
        F: FnOnce() -> TaskResult + Send + 'static,
    {
        self.submit_task(task, TaskPriority::Critical, None, None)
    }

    /// Get current scheduler statistics
    pub fn statistics(&self) -> SchedulerStats {
        let mut stats = self.scheduler_stats.lock().unwrap();

        // Update thread efficiencies
        if let Ok(thread_states) = self.thread_states.try_lock() {
            stats.thread_efficiency = thread_states
                .iter()
                .map(|state| state.efficiency())
                .collect();
        }

        stats.clone()
    }

    /// Get number of active threads
    pub fn num_threads(&self) -> usize {
        self.config.num_threads
    }

    /// Get current queue length
    pub fn queue_length(&self) -> usize {
        self.global_queue.lock().unwrap().len()
    }

    /// Shutdown the scheduler gracefully
    pub fn shutdown(&self) -> Result<()> {
        // Signal shutdown
        {
            let (shutdown_flag, condvar) = &*self.shutdown_signal;
            let mut flag = shutdown_flag.lock().unwrap();
            *flag = true;
            condvar.notify_all();
        }

        // Wait for all worker threads to finish
        // Note: We can't consume self.worker_threads here due to borrowing rules
        // In a real implementation, we'd need a different approach

        Ok(())
    }

    /// Worker thread main loop
    fn worker_thread_main(
        thread_id: usize,
        global_queue: Arc<Mutex<BinaryHeap<ScheduledTask>>>,
        shutdown_signal: Arc<(Mutex<bool>, Condvar)>,
        thread_states: Arc<Mutex<Vec<ThreadState>>>,
        scheduler_stats: Arc<Mutex<SchedulerStats>>,
        config: SchedulerConfig,
    ) {
        let (shutdown_flag, condvar) = &*shutdown_signal;

        loop {
            // Check shutdown signal
            {
                let flag = shutdown_flag.lock().unwrap();
                if *flag {
                    break;
                }
            }

            // Try to get a task from global queue
            let task = {
                let mut queue = global_queue.lock().unwrap();
                queue.pop()
            };

            match task {
                Some(scheduled_task) => {
                    let start_time = Instant::now();

                    // Execute the task
                    let result = (scheduled_task.task)();

                    let execution_time = start_time.elapsed();

                    // Update thread state
                    {
                        let mut states = thread_states.lock().unwrap();
                        if let Some(state) = states.get_mut(thread_id) {
                            state.tasks_executed += 1;
                            state.total_execution_time += execution_time;
                        }
                    }

                    // Update global statistics
                    {
                        let mut stats = scheduler_stats.lock().unwrap();
                        match result {
                            TaskResult::Success => stats.tasks_completed += 1,
                            TaskResult::Error(_) | TaskResult::Cancelled => stats.tasks_failed += 1,
                        }

                        // Update average execution time (simple moving average)
                        let total_tasks = stats.tasks_completed + stats.tasks_failed;
                        if total_tasks > 0 {
                            let old_time_nanos = stats.average_execution_time.as_nanos() as u64;
                            let new_time_nanos = execution_time.as_nanos() as u64;
                            let avg_nanos =
                                (old_time_nanos * (total_tasks - 1) + new_time_nanos) / total_tasks;
                            stats.average_execution_time = Duration::from_nanos(avg_nanos);
                        }
                    }
                }
                None => {
                    // No tasks available, wait or try work stealing
                    let idle_start = Instant::now();

                    if config.enable_adaptive_scheduling {
                        // Try work stealing from other threads
                        if Self::try_work_stealing(thread_id, &thread_states, &config) {
                            continue;
                        }
                    }

                    // Wait for notification or timeout
                    let flag = shutdown_flag.lock().unwrap();
                    if !*flag {
                        let _ =
                            condvar.wait_timeout(flag, Duration::from_millis(config.time_slice_ms));
                    }

                    let idle_time = idle_start.elapsed();
                    {
                        let mut states = thread_states.lock().unwrap();
                        if let Some(state) = states.get_mut(thread_id) {
                            state.idle_time += idle_time;
                        }
                    }
                }
            }
        }
    }

    /// Attempt work stealing from other threads
    fn try_work_stealing(
        thread_id: usize,
        thread_states: &Arc<Mutex<Vec<ThreadState>>>,
        config: &SchedulerConfig,
    ) -> bool {
        if let Ok(mut states) = thread_states.try_lock() {
            let current_time = Instant::now();

            // Check if enough time has passed since last steal attempt
            if let Some(current_state) = states.get_mut(thread_id) {
                if current_time.duration_since(current_state.last_steal_time)
                    < Duration::from_millis(config.time_slice_ms)
                {
                    return false;
                }
                current_state.last_steal_time = current_time;
            }

            // Find the thread with the most tasks
            let mut best_victim = None;
            let mut max_tasks = config.work_stealing_threshold;

            for (i, state) in states.iter().enumerate() {
                if i != thread_id && state.local_queue.len() > max_tasks {
                    max_tasks = state.local_queue.len();
                    best_victim = Some(i);
                }
            }

            // Steal from the best victim
            if let Some(victim_id) = best_victim {
                if let Some(victim_state) = states.get_mut(victim_id) {
                    if let Some(stolen_task) = victim_state.local_queue.pop() {
                        if let Some(current_state) = states.get_mut(thread_id) {
                            current_state.local_queue.push(stolen_task);
                            return true;
                        }
                    }
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_scheduler_creation() {
        let config = SchedulerConfig::optimal_for_cores(2);
        let scheduler = ParallelScheduler::new(config).unwrap();
        assert_eq!(scheduler.num_threads(), 2);
        assert_eq!(scheduler.queue_length(), 0);
    }

    #[test]
    fn test_task_submission() {
        let config = SchedulerConfig::optimal_for_cores(2);
        let scheduler = ParallelScheduler::new(config).unwrap();

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&counter);

        let task_id = scheduler
            .submit_task(
                move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    TaskResult::Success
                },
                TaskPriority::Normal,
                None,
                None,
            )
            .unwrap();

        assert!(task_id > 0);

        // Give some time for task execution
        std::thread::sleep(Duration::from_millis(100));

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_priority_scheduling() {
        let config = SchedulerConfig::optimal_for_cores(1); // Single thread for deterministic testing
        let scheduler = ParallelScheduler::new(config).unwrap();

        let execution_order = Arc::new(Mutex::new(Vec::new()));

        // Submit tasks in reverse priority order
        for priority in [
            TaskPriority::Low,
            TaskPriority::Normal,
            TaskPriority::High,
            TaskPriority::Critical,
        ] {
            let order_clone = Arc::clone(&execution_order);
            let _ = scheduler
                .submit_task(
                    move || {
                        order_clone.lock().unwrap().push(priority);
                        TaskResult::Success
                    },
                    priority,
                    None,
                    None,
                )
                .unwrap();
        }

        // Wait for execution (increased time for reliability)
        std::thread::sleep(Duration::from_millis(500));

        let order = execution_order.lock().unwrap();
        assert!(
            order.len() >= 4,
            "Expected at least 4 tasks to complete, got {}",
            order.len()
        );

        // Critical tasks should be executed first
        assert_eq!(
            order[0],
            TaskPriority::Critical,
            "Expected Critical priority task to execute first"
        );
    }

    #[test]
    fn test_scheduler_statistics() {
        let config = SchedulerConfig::optimal_for_cores(2);
        let scheduler = ParallelScheduler::new(config).unwrap();

        // Submit some tasks
        for _ in 0..5 {
            let _ = scheduler
                .submit_task(
                    || {
                        std::thread::sleep(Duration::from_millis(10));
                        TaskResult::Success
                    },
                    TaskPriority::Normal,
                    None,
                    None,
                )
                .unwrap();
        }

        // Wait for execution (increased time for reliability)
        std::thread::sleep(Duration::from_millis(500));

        let stats = scheduler.statistics();
        assert_eq!(stats.tasks_submitted, 5);
        assert!(stats.tasks_completed > 0);
        assert!(stats.thread_efficiency.len() == 2);
    }

    #[test]
    fn test_urgent_task_submission() {
        let config = SchedulerConfig::optimal_for_cores(1);
        let scheduler = ParallelScheduler::new(config).unwrap();

        let executed = Arc::new(AtomicU32::new(0));
        let executed_clone = Arc::clone(&executed);

        let task_id = scheduler
            .submit_urgent_task(move || {
                executed_clone.store(1, Ordering::SeqCst);
                TaskResult::Success
            })
            .unwrap();

        assert!(task_id > 0);

        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(executed.load(Ordering::SeqCst), 1);
    }
}
