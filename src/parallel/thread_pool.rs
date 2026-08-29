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
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// How long [`ThreadPool::wait`] sleeps between re-tests of the shutdown flag
/// while it is otherwise blocked on the completion condvar.
///
/// This is *not* a polling loop for completion -- every completion signals
/// the condvar directly (see [`CompletionTracker::record_completed`]), so a
/// waiter wakes as soon as the count moves. The timeout exists only so a
/// waiter can notice a pool that has begun shutting down underneath it; see
/// [`ThreadPool::wait`] for why that re-check is kept.
const SHUTDOWN_RECHECK_INTERVAL: Duration = Duration::from_millis(10);

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

/// Submitted/completed task counts, guarded together so a waiter can compare
/// them as one consistent observation.
#[derive(Debug, Default)]
struct TaskCounts {
    /// Tasks that have reached a queue. Monotonically increasing, except for
    /// the rollback of a submission whose enqueue failed outright.
    submitted: u64,
    /// Tasks whose closure has returned *or* unwound. Monotonically
    /// increasing, and never allowed to overtake `submitted`.
    completed: u64,
}

/// Real completion accounting behind [`ThreadPool::wait`].
///
/// `wait()` used to poll two independently-derived conditions -- the summed
/// queue lengths (`pending_tasks`) and the per-worker `is_idle` flags -- and
/// *both* can read "nothing left to do" while a task is genuinely in flight:
/// `WorkerState::pop_task` removes a task from its deque before the closure
/// runs, and `is_idle` is only ever cleared on wake-from-park, never on task
/// pickup. Between a worker popping a task and that worker being observable
/// as busy, a concurrent `wait()` saw an empty queue and an idle worker and
/// returned with work still executing.
///
/// These two counters replace that inference with a fact: only a real
/// submission moves `submitted`, only a real completion moves `completed`,
/// and both live under one mutex, so "everything submitted so far has
/// finished" is a single observation rather than a race between two.
#[derive(Debug, Default)]
struct CompletionTracker {
    counts: Mutex<TaskCounts>,
    completion: Condvar,
}

impl CompletionTracker {
    /// Lock the counts, taking the values back out of a poisoned lock.
    ///
    /// The guarded data is two integers that are only ever incremented while
    /// the lock is held, and never observed part-way through an update, so a
    /// panic elsewhere in the pool cannot leave them inconsistent. Refusing
    /// to read them after an unrelated panic would convert a recoverable
    /// situation into a permanently blocked `wait()`, which is strictly
    /// worse than continuing with counters that are still correct.
    fn lock_counts(&self) -> MutexGuard<'_, TaskCounts> {
        self.counts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Count one task as submitted.
    ///
    /// Must run *before* the task is pushed onto any queue: a worker that
    /// picked the task up first could otherwise complete it before the
    /// submission was counted, letting `completed` overtake `submitted`.
    fn record_submitted(&self) {
        self.lock_counts().submitted += 1;
    }

    /// Undo a [`CompletionTracker::record_submitted`] whose task never
    /// reached a queue.
    ///
    /// Correct *only* on a failed enqueue: in that case the task was never
    /// visible to any worker, so nothing can ever complete it, and leaving
    /// the count raised would block every later `wait()` forever.
    fn undo_submitted(&self) {
        let mut counts = self.lock_counts();
        counts.submitted = counts.submitted.saturating_sub(1);
    }

    /// Count one task as finished and wake every waiter.
    fn record_completed(&self) {
        self.lock_counts().completed += 1;
        self.completion.notify_all();
    }
}

/// Counts its task as completed when it drops.
///
/// Held *across* the task closure rather than incremented after it, so a
/// task that panics still counts on the way out. Without this, one panicking
/// closure would leave `completed` permanently short of `submitted` and
/// every subsequent [`ThreadPool::wait`] would block forever.
struct CompletionGuard<'a> {
    tracker: &'a CompletionTracker,
}

impl Drop for CompletionGuard<'_> {
    fn drop(&mut self) {
        self.tracker.record_completed();
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
    /// Submitted/completed accounting that makes [`ThreadPool::wait`] a real
    /// completion barrier rather than a guess derived from queue lengths and
    /// idle flags.
    tracker: Arc<CompletionTracker>,
    // Every worker clones this handle and pushes each finished task's id
    // (see `execute_task`), but `ThreadPool` itself never reads it back —
    // there's no public API to query completed task ids or to submit a
    // task with `PoolTask::dependencies` (always empty) gated on them.
    // Documents the dependency-tracking mechanism `PoolTask::dependencies`
    // implies; wiring both up is a scheduling-behavior change, out of
    // scope for a lint-only pass.
    #[allow(dead_code)]
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
        let tracker = Arc::new(CompletionTracker::default());

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
            let tracker_clone = Arc::clone(&tracker);
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
                    tracker_clone,
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
            tracker,
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

        // Count the task before it can be picked up. A worker that popped it
        // first would otherwise complete a task `submitted` had not yet seen,
        // letting `completed` overtake `submitted`.
        self.tracker.record_submitted();

        if let Err(e) = self.enqueue(pool_task) {
            // `enqueue` only fails before the task becomes visible to any
            // worker, so nothing will ever complete it: give the submission
            // count back, or every later `wait()` would block forever.
            self.tracker.undo_submitted();
            return Err(e);
        }

        // Update stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.tasks_submitted += 1;
        }

        Ok(task_id)
    }

    /// Hand `pool_task` to the least loaded worker, falling back to the
    /// global queue when the pool has no workers at all.
    ///
    /// Every fallible step happens strictly *before* the task becomes visible
    /// to a worker, so a returned `Err` always means "this task was not
    /// queued". That is exactly what makes the `undo_submitted` rollback in
    /// [`ThreadPool::submit_with_priority`] sound: it can never take back a
    /// submission that some worker is already running.
    fn enqueue(&self, pool_task: PoolTask) -> Result<()> {
        if let Some(worker_idx) = self.find_least_loaded_worker() {
            self.workers[worker_idx].push_task(pool_task)?;
        } else {
            let mut global = self.global_queue.lock().map_err(|_| {
                NumRs2Error::RuntimeError("Failed to acquire global queue lock".to_string())
            })?;
            global.push_back(pool_task);
        }

        self.wake_workers();
        Ok(())
    }

    /// Wake every parked worker.
    ///
    /// `notify_all`, not `notify_one`: all workers park on this one condvar,
    /// so `notify_one` may wake a worker other than the one whose deque just
    /// received the task -- and a lone queued task is not stealable
    /// (`try_steal_work` requires `queue_len() > 1`), so its owner would then
    /// sleep out a full `idle_timeout` before running it.
    ///
    /// Infallible on purpose. The mutex guards `()`, so no panic can corrupt
    /// anything behind it, and reporting an error here -- after the task is
    /// already queued -- would make the caller's rollback wrong.
    fn wake_workers(&self) {
        let (lock, cvar) = &*self.idle_notify;
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        cvar.notify_all();
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

    /// Get the configuration this pool was built with
    pub fn config(&self) -> ThreadPoolConfig {
        self.config.clone()
    }

    /// Get number of pending tasks
    pub fn pending_tasks(&self) -> usize {
        let global_count = self.global_queue.lock().map(|q| q.len()).unwrap_or(0);

        let worker_count: usize = self.workers.iter().map(|w| w.queue_len()).sum();

        global_count + worker_count
    }

    /// Wait for all tasks submitted so far to finish executing
    ///
    /// "Submitted so far" is snapshotted on entry: tasks another thread
    /// submits after this call started are deliberately not waited on, which
    /// is what stops a steady stream of submissions from making this block
    /// forever.
    ///
    /// This is a real completion barrier. It compares the pool's submitted
    /// and completed counts (see `CompletionTracker`) instead of inferring
    /// quiescence from queue lengths plus `is_idle` flags, both of which read
    /// "done" during the window between a worker popping a task and that
    /// worker becoming observably busy -- the window that let this method
    /// return with a task still running.
    ///
    /// It blocks on a condvar signalled by every completion rather than
    /// polling. The bounded re-check exists solely so a waiter cannot be
    /// wedged by a pool that starts shutting down underneath it: shutdown
    /// abandons whatever is still queued (`worker_main` re-tests the flag
    /// before every pickup), so those tasks never complete. Today
    /// [`ThreadPool::shutdown`] consumes the pool and so cannot run while a
    /// `&self` borrow sits in here, and it signals the same condvar anyway;
    /// the timeout is insurance against a future `Drop` impl or a `&self`
    /// shutdown silently turning an untimed wait into a permanent block.
    pub fn wait(&self) -> Result<()> {
        let mut counts = self.tracker.lock_counts();
        let target = counts.submitted;

        while counts.completed < target {
            if self.shutdown.load(Ordering::Relaxed) {
                break;
            }

            let (guard, _timed_out) = self
                .tracker
                .completion
                .wait_timeout(counts, SHUTDOWN_RECHECK_INTERVAL)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            counts = guard;
        }

        Ok(())
    }

    /// Shutdown the thread pool gracefully
    pub fn shutdown(self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);

        // Whatever is still queued is abandoned from here on, so its
        // completion signal will never arrive: release any waiter now
        // instead of leaving it to time out.
        self.tracker.completion.notify_all();

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
        tracker: Arc<CompletionTracker>,
        config: ThreadPoolConfig,
    ) {
        while !shutdown.load(Ordering::Relaxed) {
            let mut task_found = false;

            // 1. Try local queue
            if let Ok(Some(task)) = worker.pop_task() {
                Self::execute_task(task, &worker, &stats, &completed_tasks, &tracker);
                task_found = true;
            }

            // 2. Try global queue
            if !task_found {
                if let Ok(mut global) = global_queue.try_lock() {
                    if let Some(task) = global.pop_front() {
                        drop(global);
                        Self::execute_task(task, &worker, &stats, &completed_tasks, &tracker);
                        task_found = true;
                    }
                }
            }

            // 3. Try work stealing
            if !task_found {
                if let Some(stolen_task) = Self::try_steal_work(&worker, &workers, &config) {
                    Self::execute_task(stolen_task, &worker, &stats, &completed_tasks, &tracker);
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
        tracker: &CompletionTracker,
    ) {
        // Declared first so it drops last: whether the closure below returns
        // normally or unwinds, this task is counted exactly once, and the
        // count lands after the bookkeeping that follows the closure. Nothing
        // between the closure call and this guard's drop can panic, so a
        // panicking task cannot turn into a double panic here.
        let _completion = CompletionGuard { tracker };

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
    use std::sync::mpsc;

    /// Regression: `wait()` must not return while a task is still running.
    ///
    /// The previous implementation polled `pending_tasks() > 0 ||
    /// has_active_workers()`, two independently-derived conditions.
    /// `pop_task()` removes a task from its deque *before* the closure runs,
    /// so `pending_tasks()` reads zero while the task executes; the pool then
    /// leans on `is_idle` to cover that gap, and `is_idle` is only ever
    /// cleared on wake-from-park, never on task pickup.
    ///
    /// Repeated submit-then-wait against a single warm pool. This is the
    /// shape the barrier has to hold under, and it asserts inside the loop
    /// rather than once at the end so an early return in round 7 that later
    /// rounds "catch up" on still fails. See
    /// [`test_wait_barrier_on_a_freshly_started_pool`] for the variant that
    /// reproduces the historical failure.
    #[test]
    fn test_wait_is_a_real_completion_barrier() {
        const ROUNDS: u32 = 200;
        const TASKS_PER_ROUND: u32 = 8;

        let pool = ThreadPool::with_config(ThreadPoolConfig {
            num_threads: Some(4),
            ..Default::default()
        })
        .expect("Failed to create thread pool");
        let counter = Arc::new(AtomicU32::new(0));

        for round in 0..ROUNDS {
            for _ in 0..TASKS_PER_ROUND {
                let counter_clone = Arc::clone(&counter);
                pool.submit(move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                })
                .expect("Failed to submit task");
            }

            pool.wait().expect("Failed to wait for tasks");

            assert_eq!(
                counter.load(Ordering::SeqCst),
                (round + 1) * TASKS_PER_ROUND,
                "wait() returned in round {round} with tasks still in flight"
            );
        }
    }

    /// Regression: the exact window `test_priority_tasks` used to flake in.
    ///
    /// `WorkerState::is_idle` starts out `true` and `worker_main` clears it
    /// only on wake-from-park, so a worker that picks up a task during its
    /// very first loop iteration -- before it has ever parked -- runs that
    /// task while still reporting itself idle. Combined with `pop_task()`
    /// emptying the deque before the closure runs, the old `wait()` saw an
    /// empty queue and an all-idle pool and returned with the task not yet
    /// started. That is why the failure only ever appeared under full-suite
    /// load (which delays worker startup past the submit) and never in
    /// isolation.
    ///
    /// Reproduced deterministically by submitting into a pool that was just
    /// constructed, many times over. The closures sleep so "popped but not
    /// finished" is an observable interval rather than a few hundred
    /// nanoseconds, and the counter is read *before* `shutdown()` -- joining
    /// the workers first would let an in-flight straggler finish and hide
    /// exactly the bug under test.
    #[test]
    fn test_wait_barrier_on_a_freshly_started_pool() {
        const ROUNDS: u32 = 200;
        const TASKS_PER_ROUND: u32 = 4;

        for round in 0..ROUNDS {
            let pool = ThreadPool::with_config(ThreadPoolConfig {
                num_threads: Some(2),
                ..Default::default()
            })
            .expect("Failed to create thread pool");
            let counter = Arc::new(AtomicU32::new(0));

            for _ in 0..TASKS_PER_ROUND {
                let counter_clone = Arc::clone(&counter);
                pool.submit(move || {
                    thread::sleep(Duration::from_micros(500));
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                })
                .expect("Failed to submit task");
            }

            pool.wait().expect("Failed to wait for tasks");
            let completed_at_barrier = counter.load(Ordering::SeqCst);

            // Consumes the pool and joins its workers, so 200 rounds do not
            // leave 400 detached threads behind.
            pool.shutdown().expect("Failed to shut down thread pool");

            assert_eq!(
                completed_at_barrier, TASKS_PER_ROUND,
                "wait() returned in round {round} with tasks still in flight"
            );
        }
    }

    /// A task that panics must still count as completed.
    ///
    /// Completion is recorded by a drop guard held across the closure, so an
    /// unwinding task is counted on its way out. Without that guard the
    /// pool's completed count would stay permanently one short of its
    /// submitted count and *every* later `wait()` would block forever.
    ///
    /// `wait()` runs on a helper thread reporting through a channel so the
    /// missing-guard regression fails on the `recv_timeout` instead of
    /// hanging the suite. Nothing is submitted after the panic: the
    /// panicking worker thread is gone, and `find_least_loaded_worker` would
    /// happily hand a follow-up task to its now-empty deque.
    #[test]
    fn test_wait_returns_after_a_panicking_task() {
        let pool = Arc::new(
            ThreadPool::with_config(ThreadPoolConfig {
                num_threads: Some(2),
                ..Default::default()
            })
            .expect("Failed to create thread pool"),
        );

        pool.submit(|| panic!("intentional panic exercising completion accounting"))
            .expect("Failed to submit task");

        let waiter_pool = Arc::clone(&pool);
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let result = waiter_pool.wait();
            let _ = tx.send(result.is_ok());
        });

        let waited = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("wait() never returned after a panicking task");
        assert!(waited, "wait() reported an error after a panicking task");
    }

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
