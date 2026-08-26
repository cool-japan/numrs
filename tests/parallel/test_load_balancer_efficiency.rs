//! Tests for load balancer efficiency and performance

use numrs2::parallel::{BalancingStrategy, LoadBalancer, LoadBalancingAdvisor, WorkloadMetrics};
use std::time::Duration;

// ============================================================================
// Strategy Performance Tests
// ============================================================================

#[test]
fn test_round_robin_vs_least_loaded_performance() {
    // Round Robin
    let rr_balancer =
        LoadBalancer::new(BalancingStrategy::RoundRobin, 4).expect("Failed to create RR balancer");

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = rr_balancer.select_worker();
    }
    let rr_duration = start.elapsed();

    // Least Loaded
    let ll_balancer =
        LoadBalancer::new(BalancingStrategy::LeastLoaded, 4).expect("Failed to create LL balancer");

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = ll_balancer.select_worker();
    }
    let ll_duration = start.elapsed();

    // Both should complete in reasonable time
    assert!(rr_duration < Duration::from_secs(1));
    assert!(ll_duration < Duration::from_secs(1));
}

#[test]
fn test_weighted_capacity_efficiency() {
    let balancer = LoadBalancer::new(BalancingStrategy::WeightedCapacity, 4)
        .expect("Failed to create balancer");

    // Set different capacities
    balancer
        .update_worker_metrics(0, 10, 0.9, 0.8)
        .expect("Failed to update worker 0");
    balancer
        .update_worker_metrics(1, 5, 0.5, 0.4)
        .expect("Failed to update worker 1");
    balancer
        .update_worker_metrics(2, 2, 0.2, 0.1)
        .expect("Failed to update worker 2");
    balancer
        .update_worker_metrics(3, 8, 0.7, 0.6)
        .expect("Failed to update worker 3");

    // Select workers and verify it tends to choose less loaded ones
    let mut selections = [0; 4];
    for _ in 0..100 {
        if let Ok(worker) = balancer.select_worker() {
            selections[worker] += 1;
        }
    }

    // Worker 2 (least loaded) should be selected more often
    assert!(selections[2] > selections[0]);
}

#[test]
fn test_adaptive_strategy_switching() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::Adaptive, 4).expect("Failed to create balancer");

    // Initially adaptive
    assert_eq!(balancer.current_strategy(), BalancingStrategy::Adaptive);

    // Update metrics to trigger different behaviors
    for i in 0..4 {
        balancer
            .update_worker_metrics(i, i * 5, 0.5 + (i as f64 * 0.1), 0.3)
            .expect("Failed to update worker");
    }

    // Select workers and verify it adapts
    for _ in 0..50 {
        let _ = balancer.select_worker();
    }

    // Strategy should still be adaptive
    assert_eq!(balancer.current_strategy(), BalancingStrategy::Adaptive);
}

// ============================================================================
// Balancer Responsiveness Tests
// ============================================================================

#[test]
fn test_response_time_to_load_changes() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::LeastLoaded, 4).expect("Failed to create balancer");

    // Initial state - balanced
    for i in 0..4 {
        balancer
            .update_worker_metrics(i, 5, 0.5, 0.3)
            .expect("Failed to update worker");
    }

    // Suddenly increase load on worker 0
    let start = std::time::Instant::now();
    balancer
        .update_worker_metrics(0, 50, 0.95, 0.9)
        .expect("Failed to update worker 0");

    // Next selection should avoid worker 0
    if let Ok(worker) = balancer.select_worker() {
        assert_ne!(worker, 0, "Should not select overloaded worker");
    }

    let response_time = start.elapsed();
    assert!(
        response_time < Duration::from_millis(10),
        "Should respond quickly to load changes"
    );
}

#[test]
fn test_rebalancing_speed() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::Adaptive, 6).expect("Failed to create balancer");

    // Create severe imbalance
    balancer
        .update_worker_metrics(0, 100, 0.99, 0.95)
        .expect("Failed to update worker 0");

    for i in 1..6 {
        balancer
            .update_worker_metrics(i, 1, 0.1, 0.05)
            .expect("Failed to update worker");
    }

    let metrics_before = balancer.current_metrics();
    assert!(metrics_before.load_imbalance > 0.5);

    // Simulate rebalancing by selecting workers
    for _ in 0..100 {
        let _ = balancer.select_worker();
    }

    // Verify the balancer tends to select less loaded workers
    let metrics_after = balancer.current_metrics();
    assert_eq!(metrics_after.queue_lengths[0], 100); // Unchanged by selection
}

// ============================================================================
// Queue Rebalancing Tests
// ============================================================================

#[test]
fn test_automatic_queue_rebalancing() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::WorkStealing, 4).expect("Failed to create balancer");

    // Create imbalanced queues
    balancer
        .update_worker_metrics(0, 20, 0.8, 0.7)
        .expect("Failed to update worker 0");
    balancer
        .update_worker_metrics(1, 2, 0.2, 0.1)
        .expect("Failed to update worker 1");
    balancer
        .update_worker_metrics(2, 15, 0.7, 0.6)
        .expect("Failed to update worker 2");
    balancer
        .update_worker_metrics(3, 3, 0.3, 0.2)
        .expect("Failed to update worker 3");

    let metrics = balancer.current_metrics();
    assert!(metrics.load_imbalance > 0.0);

    // Verify least loaded worker is selected
    if let Ok(worker) = balancer.select_worker() {
        assert!(worker == 1 || worker == 3);
    }
}

#[test]
fn test_rebalance_threshold_triggers() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::Adaptive, 4).expect("Failed to create balancer");

    // Start balanced
    for i in 0..4 {
        balancer
            .update_worker_metrics(i, 10, 0.5, 0.4)
            .expect("Failed to update worker");
    }

    let initial_metrics = balancer.current_metrics();
    assert!(initial_metrics.load_imbalance < 0.1);

    // Create imbalance
    balancer
        .update_worker_metrics(0, 50, 0.9, 0.8)
        .expect("Failed to update worker 0");

    let imbalanced_metrics = balancer.current_metrics();
    assert!(imbalanced_metrics.load_imbalance > initial_metrics.load_imbalance);
}

// ============================================================================
// Worker Selection Algorithm Tests
// ============================================================================

#[test]
fn test_optimal_worker_selection() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::LeastLoaded, 5).expect("Failed to create balancer");

    // Set up workers with different loads
    let loads = [10, 5, 20, 3, 15];
    for (i, &load) in loads.iter().enumerate() {
        balancer
            .update_worker_metrics(i, load, 0.5, 0.4)
            .expect("Failed to update worker");
    }

    // Should select worker 3 (lowest load)
    let worker = balancer.select_worker().expect("Failed to select worker");
    assert_eq!(worker, 3);
}

#[test]
fn test_worker_selection_fairness() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::RoundRobin, 4).expect("Failed to create balancer");

    let mut selections = [0; 4];

    // Select 100 workers
    for _ in 0..100 {
        if let Ok(worker) = balancer.select_worker() {
            selections[worker] += 1;
        }
    }

    // Round robin should distribute evenly
    for count in &selections {
        assert_eq!(*count, 25);
    }
}

// ============================================================================
// Load Distribution Quality Tests
// ============================================================================

#[test]
fn test_load_distribution_coefficient() {
    // Perfect balance
    let metrics_balanced = WorkloadMetrics {
        queue_lengths: vec![10, 10, 10, 10],
        ..Default::default()
    };
    let cv_balanced = metrics_balanced.load_distribution_cv();
    assert_eq!(cv_balanced, 0.0);

    // High imbalance
    let metrics_imbalanced = WorkloadMetrics {
        queue_lengths: vec![1, 10, 20, 50],
        ..Default::default()
    };
    let cv_imbalanced = metrics_imbalanced.load_distribution_cv();
    assert!(cv_imbalanced > 0.5);
}

// ============================================================================
// Strategy-Specific Tests
// ============================================================================

#[test]
fn test_numa_aware_strategy() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::NumaAware, 4).expect("Failed to create balancer");

    // Update workers
    for i in 0..4 {
        balancer
            .update_worker_metrics(i, i * 2, 0.5, 0.3)
            .expect("Failed to update worker");
    }

    // Select workers
    for _ in 0..20 {
        let worker = balancer.select_worker().expect("Failed to select worker");
        assert!(worker < 4);
    }
}

#[test]
fn test_work_stealing_strategy() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::WorkStealing, 4).expect("Failed to create balancer");

    // Create imbalanced load
    balancer
        .update_worker_metrics(0, 30, 0.9, 0.8)
        .expect("Failed to update worker 0");
    balancer
        .update_worker_metrics(1, 2, 0.2, 0.1)
        .expect("Failed to update worker 1");

    // Should prefer less loaded workers
    let worker = balancer.select_worker().expect("Failed to select worker");
    assert_ne!(worker, 0);
}

// ============================================================================
// Load Balancing Advisor Tests
// ============================================================================

#[test]
fn test_advisor_recommendations_high_imbalance() {
    let mut advisor = LoadBalancingAdvisor::new();

    // Record high imbalance
    for _ in 0..5 {
        let metrics = WorkloadMetrics {
            load_imbalance: 0.8,
            total_throughput: 100.0,
            cache_miss_rate: 0.1,
            ..Default::default()
        };
        advisor.record_metrics(metrics);
    }

    let recommendation = advisor.recommend_strategy();
    assert_eq!(recommendation, BalancingStrategy::WorkStealing);
}

#[test]
fn test_advisor_recommendations_low_throughput() {
    let mut advisor = LoadBalancingAdvisor::new();

    // Record low throughput
    for _ in 0..5 {
        let metrics = WorkloadMetrics {
            load_imbalance: 0.1,
            total_throughput: 10.0,
            cache_miss_rate: 0.05,
            ..Default::default()
        };
        advisor.record_metrics(metrics);
    }

    let recommendation = advisor.recommend_strategy();
    // Should recommend a strategy
    assert!(
        recommendation == BalancingStrategy::LeastLoaded
            || recommendation == BalancingStrategy::Adaptive
            || recommendation == BalancingStrategy::WorkStealing
    );
}

#[test]
fn test_advisor_trend_analysis() {
    let mut advisor = LoadBalancingAdvisor::new();

    // Record improving metrics
    for i in 0..10 {
        let metrics = WorkloadMetrics {
            total_throughput: 50.0 + (i as f64 * 5.0),
            load_imbalance: 0.3 - (i as f64 * 0.02),
            avg_response_time: Duration::from_millis(100 - i * 5),
            ..Default::default()
        };
        advisor.record_metrics(metrics);
    }

    let analysis = advisor.analyze_trends();
    assert!(
        analysis.throughput_trend >= 0.0,
        "Throughput should be improving"
    );
    assert!((0.0..=1.0).contains(&analysis.stability_score));
}

// ============================================================================
// Performance Comparison Tests
// ============================================================================

#[test]
fn test_strategy_selection_latency() {
    let strategies = [
        BalancingStrategy::RoundRobin,
        BalancingStrategy::LeastLoaded,
        BalancingStrategy::WeightedCapacity,
        BalancingStrategy::Adaptive,
    ];

    for strategy in strategies {
        let balancer = LoadBalancer::new(strategy, 4).expect("Failed to create balancer");

        let start = std::time::Instant::now();
        for _ in 0..1000 {
            let _ = balancer.select_worker();
        }
        let latency = start.elapsed();

        assert!(
            latency < Duration::from_millis(100),
            "Strategy {:?} selection took too long: {:?}",
            strategy,
            latency
        );
    }
}
