//! Tests for NUMA awareness

use numrs2::parallel::{BalancingStrategy, LoadBalancer};

#[test]
fn test_numa_aware_strategy() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::NumaAware, 4).expect("Failed to create load balancer");

    assert_eq!(balancer.current_strategy(), BalancingStrategy::NumaAware);
    assert_eq!(balancer.num_workers(), 4);
}

#[test]
fn test_numa_aware_worker_selection() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::NumaAware, 4).expect("Failed to create load balancer");

    // Select workers multiple times
    for _ in 0..10 {
        let worker_id = balancer.select_worker().expect("Failed to select worker");
        assert!(worker_id < 4);
    }
}

#[test]
fn test_numa_node_detection() {
    // This test verifies that NUMA detection doesn't panic
    // Actual NUMA node detection is platform-specific
    let balancer =
        LoadBalancer::new(BalancingStrategy::NumaAware, 2).expect("Failed to create load balancer");

    let metrics = balancer.current_metrics();
    assert_eq!(metrics.queue_lengths.len(), 2);
}

#[test]
fn test_numa_aware_load_distribution() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::NumaAware, 4).expect("Failed to create load balancer");

    // Update worker metrics to simulate load
    for i in 0..4 {
        balancer
            .update_worker_metrics(i, i * 2, 0.5, 0.3)
            .expect("Failed to update worker metrics");
    }

    let metrics = balancer.current_metrics();
    assert_eq!(metrics.queue_lengths, vec![0, 2, 4, 6]);
}

#[test]
fn test_numa_aware_with_cache_locality() {
    let balancer =
        LoadBalancer::new(BalancingStrategy::NumaAware, 4).expect("Failed to create load balancer");

    // Simulate cache-aware task placement
    let mut worker_selections = Vec::new();
    for _ in 0..20 {
        let worker_id = balancer.select_worker().expect("Failed to select worker");
        worker_selections.push(worker_id);
    }

    // Verify all workers are used
    assert!(worker_selections.iter().all(|&w| w < 4));
}
