//! Comprehensive tests for parallel computing capabilities

pub mod test_adaptive_scheduling;
pub mod test_load_balancer_efficiency;
pub mod test_load_balancing;
pub mod test_metrics_monitoring;
pub mod test_numa_awareness;
pub mod test_parallel_algorithms;
pub mod test_scalability;
pub mod test_scheduler_granularity;
pub mod test_stress;
pub mod test_thread_affinity;
pub mod test_work_stealing;
pub mod test_work_stealing_advanced;

use numrs2::parallel::WorkStealingPool;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Poll a [`WorkStealingPool`] until it reports its queues empty (no
/// pending tasks, no worker mid-search-for-work), or `timeout` elapses.
///
/// This is a "queues empty" signal, not a "all submitted work has actually
/// finished executing" signal -- see [`wait_for_count`] for why, and use
/// that (or an equivalent poll on real, observable completion) for any
/// assertion that depends on every submitted closure having already run.
/// `WorkStealingPool` (unlike `ThreadPool`) exposes no blocking `wait()` at
/// all, so this at least avoids gambling that one fixed `thread::sleep` was
/// long enough for the queues themselves to empty out. Bounded by `timeout`
/// so a real regression fails the assertion that follows instead of
/// hanging the test.
fn wait_for_drain(pool: &WorkStealingPool, timeout: Duration) {
    let start = Instant::now();
    while (pool.pending_tasks() > 0 || pool.active_workers() > 0) && start.elapsed() < timeout {
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Poll `counter` until it reaches `expected`, then assert equality.
///
/// Neither `ThreadPool::wait()` nor [`wait_for_drain`] is a true completion
/// signal: `pop_task()` removes a task from its queue *before* the task's
/// closure runs, so `pending_tasks()` (and thus both pools' "am I drained"
/// check) can read zero while the last task per worker is still inside its
/// closure. Separately, both pools' `is_idle` flag is only ever cleared on
/// wake-from-a-parked-wait, never on successful task pickup, so it reads
/// "idle" throughout an entire first burst of back-to-back task execution
/// -- `active_workers()`/`has_active_workers()` cannot fill the gap either.
/// Concretely, both can return with up to one task per worker still
/// executing, which a sleeping closure makes an observable-sized window.
/// Tests that submit closures containing a `thread::sleep` and then assert
/// an exact resulting count must poll the actual side effect (this
/// function), not just synchronize on the pool and assert immediately. A
/// genuinely lost task still fails this, just after `timeout` instead of
/// immediately, with the same left/right diagnostic.
fn wait_for_count(counter: &AtomicU32, expected: u32, timeout: Duration) {
    let start = Instant::now();
    while counter.load(Ordering::SeqCst) < expected && start.elapsed() < timeout {
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(counter.load(Ordering::SeqCst), expected);
}
