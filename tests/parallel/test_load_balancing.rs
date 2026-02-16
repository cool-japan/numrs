//! Tests for load balancing

use numrs2::parallel::{
    BalancingStrategy, LoadBalancer, LoadBalancingAdvisor, WorkloadMetrics,
};
use std::time::Duration;

#[test]
fn test_round_robin_balancing() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::RoundRobin, 3).expect("Failed to create load balancer");

    let selections: Vec<usize> = (0..6)
        .map(|_| balancer.select_worker().expect("Failed to select worker"))
        .collect();

    assert_eq!(selections, vec![0, 1, 2, 0, 1, 2]);
}

#[test]
fn test_least_loaded_balancing() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::LeastLoaded, 3).expect("Failed to create load balancer");

    // Update worker 1 to be heavily loaded
    balancer
        .update_worker_metrics(1, 10, 0.8, 0.7)
        .expect("Failed to update worker metrics");

    // Select worker should avoid worker 1
    let worker = balancer.select_worker().expect("Failed to select worker");
    assert!(worker < 3);
}

#[test]
fn test_weighted_capacity_balancing() {
    let balancer = LoadBalancer::new(BalancingStrategy::WeightedCapacity, 4)
        .expect("Failed to create load balancer");

    // Update workers with different loads
    for i in 0..4 {
        balancer
            .update_worker_metrics(i, i * 3, 0.5, 0.4)
            .expect("Failed to update worker metrics");
    }

    let worker = balancer.select_worker().expect("Failed to select worker");
    assert!(worker < 4);
}

#[test]
fn test_adaptive_balancing() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::Adaptive, 4).expect("Failed to create load balancer");

    // Update metrics to trigger adaptation
    for i in 0..4 {
        balancer
            .update_worker_metrics(i, i * 2, 0.6, 0.5)
            .expect("Failed to update worker metrics");
    }

    let worker = balancer.select_worker().expect("Failed to select worker");
    assert!(worker < 4);
}

#[test]
fn test_strategy_switching() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::RoundRobin, 2).expect("Failed to create load balancer");

    assert_eq!(balancer.current_strategy(), BalancingStrategy::RoundRobin);

    balancer.set_strategy(BalancingStrategy::LeastLoaded);
    assert_eq!(balancer.current_strategy(), BalancingStrategy::LeastLoaded);
}

#[test]
fn test_workload_metrics() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::LeastLoaded, 3).expect("Failed to create load balancer");

    // Update workers
    balancer
        .update_worker_metrics(0, 5, 0.5, 0.4)
        .expect("Failed to update worker 0");
    balancer
        .update_worker_metrics(1, 3, 0.4, 0.3)
        .expect("Failed to update worker 1");
    balancer
        .update_worker_metrics(2, 7, 0.6, 0.5)
        .expect("Failed to update worker 2");

    let metrics = balancer.current_metrics();
    assert_eq!(metrics.active_tasks, 15);
    assert_eq!(metrics.queue_lengths, vec![5, 3, 7]);
    assert!(metrics.load_imbalance >= 0.0 && metrics.load_imbalance <= 1.0);
}

#[test]
fn test_load_distribution_coefficient() {
    let metrics = WorkloadMetrics {
        queue_lengths: vec![5, 5, 5],
        ..Default::default()
    };
    assert_eq!(metrics.load_distribution_cv(), 0.0);

    let metrics2 = WorkloadMetrics {
        queue_lengths: vec![1, 5, 9],
        ..Default::default()
    };
    assert!(metrics2.load_distribution_cv() > 0.5);
}

#[test]
fn test_load_balancing_advisor() {
    let mut advisor = LoadBalancingAdvisor::new();

    // Record metrics with high imbalance
    let metrics = WorkloadMetrics {
        load_imbalance: 0.5,
        total_throughput: 10.0,
        cache_miss_rate: 0.05,
        ..Default::default()
    };

    advisor.record_metrics(metrics);

    let recommendation = advisor.recommend_strategy();
    assert_eq!(recommendation, BalancingStrategy::WorkStealing);
}

#[test]
fn test_advisor_trend_analysis() {
    let mut advisor = LoadBalancingAdvisor::new();

    // Record multiple metrics
    for i in 0..10 {
        let metrics = WorkloadMetrics {
            total_throughput: 10.0 + i as f64,
            load_imbalance: 0.1,
            avg_response_time: Duration::from_millis(100),
            ..Default::default()
        };
        advisor.record_metrics(metrics);
    }

    let analysis = advisor.analyze_trends();
    assert!(analysis.throughput_trend >= 0.0);
    assert!(analysis.stability_score >= 0.0 && analysis.stability_score <= 1.0);
}

#[test]
fn test_workload_metrics_helpers() {
    let metrics = WorkloadMetrics {
        queue_lengths: vec![1, 5, 3, 7, 2],
        load_imbalance: 0.857,
        ..Default::default()
    };

    assert_eq!(metrics.most_loaded_worker(), Some(3));
    assert_eq!(metrics.least_loaded_worker(), Some(0));
    assert!(!metrics.is_balanced(0.3));
    assert!(metrics.is_balanced(0.9));
}
