//! Regression tests for LANE W1-K item 7: fabricated telemetry.
//!
//! - `src/parallel/load_balancer.rs`: `WorkloadMetrics::avg_response_time`
//!   used to be a hardcoded `Duration::from_millis(100)` regardless of
//!   actual worker behavior. It is now computed from real per-worker
//!   task-completion timings recorded via
//!   `LoadBalancer::record_task_completion` (previously-dead
//!   `tasks_completed` / `total_execution_time` fields that were tracked
//!   but never fed).
//! - `src/memory_alloc/performance_tuning.rs`:
//!   `has_consistent_allocation_sizes` used to unconditionally return
//!   `true` (making `UsePoolAllocation` fire for *any* workload once the
//!   sample-size gate passed). It is now computed from the real allocation
//!   size distribution (`record_allocation` now tracks a running
//!   mean/variance via Welford's algorithm), so it fires for genuinely
//!   consistent sizes and not for wildly varying ones.

use numrs2::memory_alloc::performance_tuning::{OptimizationType, PerformanceTuner, TuningConfig};
use numrs2::parallel::load_balancer::{BalancingStrategy, LoadBalancer};
use std::time::Duration;

// ---------------------------------------------------------------------
// load_balancer: avg_response_time
// ---------------------------------------------------------------------

#[test]
fn avg_response_time_is_zero_before_any_task_completion_recorded() {
    let balancer = LoadBalancer::new(BalancingStrategy::LeastLoaded, 2)
        .expect("failed to create load balancer");

    // Honest default: no completions recorded yet, so there is no real
    // response-time data -- this must read as zero, not a fabricated
    // fixed duration.
    let metrics = balancer.current_metrics();
    assert_eq!(metrics.avg_response_time, Duration::ZERO);
}

#[test]
fn avg_response_time_reflects_real_recorded_task_completions() {
    let balancer = LoadBalancer::new(BalancingStrategy::LeastLoaded, 2)
        .expect("failed to create load balancer");

    // Worker 0: two completions totalling 300ms.
    balancer
        .record_task_completion(0, Duration::from_millis(100))
        .expect("worker 0 exists");
    balancer
        .record_task_completion(0, Duration::from_millis(200))
        .expect("worker 0 exists");
    // Worker 1: one completion of 50ms.
    balancer
        .record_task_completion(1, Duration::from_millis(50))
        .expect("worker 1 exists");

    // Weighted average across all workers: (100 + 200 + 50) ms / 3 tasks
    // = 350/3 ms ~= 116.67ms. This must be a real computed value, not the
    // old fabricated `Duration::from_millis(100)` placeholder (which this
    // scenario would not coincidentally match).
    let metrics = balancer.current_metrics();
    let millis = metrics.avg_response_time.as_millis();
    assert!(
        (116..=117).contains(&millis),
        "expected avg_response_time ~= 116.67ms from real recorded completions, got {:?}",
        metrics.avg_response_time
    );
}

#[test]
fn record_task_completion_errs_on_invalid_worker_id() {
    let balancer = LoadBalancer::new(BalancingStrategy::LeastLoaded, 2)
        .expect("failed to create load balancer");
    let result = balancer.record_task_completion(99, Duration::from_millis(10));
    assert!(result.is_err());
}

// ---------------------------------------------------------------------
// performance_tuning: has_consistent_allocation_sizes (via
// analyze_performance -> UsePoolAllocation, since the check is private).
// ---------------------------------------------------------------------

#[test]
fn pool_allocation_recommended_for_genuinely_consistent_sizes() {
    let tuner = PerformanceTuner::new(TuningConfig {
        min_sample_size: 100,
        ..TuningConfig::default()
    });

    // 150 allocations, every one exactly 128 bytes: zero variance, the most
    // consistent case possible.
    for _ in 0..150 {
        tuner.record_allocation(128, Duration::from_nanos(500));
    }

    let recommendations = tuner.analyze_performance();
    let has_pool_recommendation = recommendations
        .iter()
        .any(|r| r.optimization_type == OptimizationType::UsePoolAllocation);
    assert!(
        has_pool_recommendation,
        "uniform allocation sizes should be flagged as pool-allocation candidates"
    );
}

#[test]
fn pool_allocation_not_recommended_for_wildly_inconsistent_sizes() {
    let tuner = PerformanceTuner::new(TuningConfig {
        min_sample_size: 100,
        ..TuningConfig::default()
    });

    // 150 allocations alternating between a tiny size and a huge one:
    // coefficient of variation is close to 1.0, nowhere near "consistent".
    // This is the scenario that a hardcoded `true` would have gotten wrong.
    for i in 0..150 {
        let size = if i % 2 == 0 { 10 } else { 100_000 };
        tuner.record_allocation(size, Duration::from_nanos(500));
    }

    let recommendations = tuner.analyze_performance();
    let has_pool_recommendation = recommendations
        .iter()
        .any(|r| r.optimization_type == OptimizationType::UsePoolAllocation);
    assert!(
        !has_pool_recommendation,
        "wildly varying allocation sizes must not be flagged as pool-allocation candidates"
    );
}

#[test]
fn simd_alignment_never_fabricated_from_timing_alone() {
    let tuner = PerformanceTuner::new(TuningConfig {
        min_sample_size: 10,
        ..TuningConfig::default()
    });

    // Slow, consistent allocations (well over both the 5_000ns and
    // 10_000ns thresholds `analyze_timing_performance` checks), but with no
    // SIMD signal available. `OptimizeAlignment` must not appear: this
    // tuner has no honest way to know the workload is SIMD-intensive from
    // allocation telemetry alone (see `has_simd_workload`'s doc comment).
    for _ in 0..20 {
        tuner.record_allocation(64, Duration::from_nanos(20_000));
    }

    let recommendations = tuner.analyze_performance();
    let has_simd_recommendation = recommendations
        .iter()
        .any(|r| r.optimization_type == OptimizationType::OptimizeAlignment);
    assert!(
        !has_simd_recommendation,
        "OptimizeAlignment must not fire without a real SIMD-workload signal"
    );
}
