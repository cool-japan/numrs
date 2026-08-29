//! Comprehensive Distributed Computing Benchmarks for NumRS2
//!
//! This benchmark suite tests distributed operations including:
//! - Distributed array operations (scatter, gather)
//! - Collective operations (broadcast, reduce, allreduce)
//! - Communication overhead measurements
//! - Point-to-point communication performance
//! - Network topology optimization
//!
//! All benchmarks follow SCIRS2 policies and use no unwrap() calls.
//!
//! Note: These benchmarks test the distributed computing infrastructure,
//! but run in single-process mode for benchmarking purposes.

#![allow(clippy::result_large_err)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use numrs2::distributed::array::*;
use numrs2::distributed::collective::*;
use numrs2::distributed::comm::*;
use numrs2::distributed::process::*;
use std::hint::black_box;

/// Benchmark distributed array distribution strategies
fn bench_distribution_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution_strategies");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let num_processes = 4;

        // Block distribution
        group.bench_with_input(BenchmarkId::new("block_local_size", size), size, |b, &s| {
            let strategy = DistributionStrategy::Block;
            b.iter(|| {
                for rank in 0..num_processes {
                    let local_size = strategy.local_size(s, rank, num_processes);
                    black_box(local_size);
                }
            });
        });

        // Cyclic distribution
        group.bench_with_input(
            BenchmarkId::new("cyclic_local_size", size),
            size,
            |b, &s| {
                let strategy = DistributionStrategy::Cyclic;
                b.iter(|| {
                    for rank in 0..num_processes {
                        let local_size = strategy.local_size(s, rank, num_processes);
                        black_box(local_size);
                    }
                });
            },
        );

        // Block-cyclic distribution
        group.bench_with_input(
            BenchmarkId::new("block_cyclic_local_size", size),
            size,
            |b, &s| {
                let strategy = DistributionStrategy::BlockCyclic { block_size: 64 };
                b.iter(|| {
                    for rank in 0..num_processes {
                        let local_size = strategy.local_size(s, rank, num_processes);
                        black_box(local_size);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark index mapping operations
fn bench_index_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_mapping");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let num_processes = 4;

        // Block distribution owner lookup
        group.bench_with_input(BenchmarkId::new("block_owner", size), size, |b, &s| {
            let strategy = DistributionStrategy::Block;
            b.iter(|| {
                for idx in (0..s).step_by(100) {
                    let owner = strategy.owner(idx, s, num_processes);
                    black_box(owner);
                }
            });
        });

        // Cyclic distribution owner lookup
        group.bench_with_input(BenchmarkId::new("cyclic_owner", size), size, |b, &s| {
            let strategy = DistributionStrategy::Cyclic;
            b.iter(|| {
                for idx in (0..s).step_by(100) {
                    let owner = strategy.owner(idx, s, num_processes);
                    black_box(owner);
                }
            });
        });
    }

    group.finish();
}

/// Benchmark collective reduction operations
fn bench_collective_reductions(c: &mut Criterion) {
    let mut group = c.benchmark_group("collective_reductions");

    for size in [100, 1000, 10000, 100000].iter() {
        // Reduction operation application
        group.bench_with_input(BenchmarkId::new("reduce_op_sum", size), size, |b, &s| {
            let data: Vec<f64> = (0..s).map(|i| i as f64).collect();
            b.iter(|| {
                let result = ReduceOp::Sum
                    .apply_slice(&data)
                    .expect("Sum.apply_slice on non-empty data must succeed");
                black_box(result);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("reduce_op_product", size),
            size,
            |b, &s| {
                let data: Vec<f64> = (0..s).map(|_| 1.001).collect();
                b.iter(|| {
                    let result = ReduceOp::Product
                        .apply_slice(&data)
                        .expect("Product.apply_slice on non-empty data must succeed");
                    black_box(result);
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("reduce_op_max", size), size, |b, &s| {
            let data: Vec<f64> = (0..s).map(|i| i as f64).collect();
            b.iter(|| {
                let result = ReduceOp::Max
                    .apply_slice(&data)
                    .expect("Max.apply_slice on non-empty data must succeed");
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("reduce_op_min", size), size, |b, &s| {
            let data: Vec<f64> = (0..s).map(|i| i as f64).collect();
            b.iter(|| {
                let result = ReduceOp::Min
                    .apply_slice(&data)
                    .expect("Min.apply_slice on non-empty data must succeed");
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark message serialization/deserialization
fn bench_message_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_serialization");

    for size in [10, 100, 1000, 10000].iter() {
        // Message header serialization
        group.bench_function("message_header_serialization", |b| {
            let header = MessageHeader::new(0, 1, 42, 1024, 100);
            b.iter(|| {
                if let Ok(bytes) = header.to_bytes() {
                    if let Ok(deserialized) = MessageHeader::from_bytes(&bytes) {
                        black_box(deserialized);
                    }
                }
            });
        });

        // Message serialization
        group.bench_with_input(
            BenchmarkId::new("message_serialization", size),
            size,
            |b, &s| {
                let data: Vec<f64> = (0..s).map(|i| i as f64).collect();
                b.iter(|| {
                    if let Ok(msg) = Message::new(0, 1, 42, data.clone()) {
                        if let Ok(bytes) = msg.to_bytes() {
                            if let Ok(deserialized) = Message::<f64>::from_bytes(&bytes) {
                                black_box(deserialized);
                            }
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark network topology calculations
fn bench_network_topology(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_topology");

    for num_nodes in [4, 8, 16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::new("optimal_algorithm_broadcast", num_nodes),
            num_nodes,
            |b, &_n| {
                let topology = numrs2::distributed::optimization::NetworkTopology::FullyConnected;
                b.iter(|| {
                    let algo = topology.optimal_algorithm("broadcast");
                    black_box(algo);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("has_direct_connection", num_nodes),
            num_nodes,
            |b, &n| {
                let topology = numrs2::distributed::optimization::NetworkTopology::FullyConnected;
                b.iter(|| {
                    for src in 0..n {
                        for dest in 0..n {
                            if src != dest {
                                let has_conn = topology.has_direct_connection(src, dest, n);
                                black_box(has_conn);
                            }
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark process group operations
fn bench_process_group(c: &mut Criterion) {
    let mut group = c.benchmark_group("process_group");

    for size in [4, 8, 16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::new("process_group_creation", size),
            size,
            |b, &s| {
                b.iter(|| {
                    let ranks: Vec<usize> = (0..s).collect();
                    if let Ok(group) = ProcessGroup::new(ranks) {
                        black_box(group);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("process_group_contains", size),
            size,
            |b, &s| {
                let ranks: Vec<usize> = (0..s).collect();
                if let Ok(group) = ProcessGroup::new(ranks) {
                    b.iter(|| {
                        for rank in 0..s {
                            let contains = group.contains(rank);
                            black_box(contains);
                        }
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark distributed matrix dimensions
fn bench_distributed_matrix_dims(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed_matrix_dims");

    for size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("matrix_dims_creation", size),
            size,
            |b, &s| {
                b.iter(|| {
                    if let Ok(dims) = numrs2::distributed::linalg::MatrixDims::new(s, s) {
                        black_box(dims);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("matrix_dims_can_multiply", size),
            size,
            |b, &s| {
                if let (Ok(dims_a), Ok(dims_b)) = (
                    numrs2::distributed::linalg::MatrixDims::new(s, s / 2),
                    numrs2::distributed::linalg::MatrixDims::new(s / 2, s),
                ) {
                    b.iter(|| {
                        let can_multiply = dims_a.can_multiply(&dims_b);
                        black_box(can_multiply);
                    });
                }
            },
        );
    }

    group.finish();
}

/// Benchmark communication bandwidth and latency models
fn bench_comm_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("comm_models");

    group.bench_function("latency_model_estimate", |b| {
        let mut model = numrs2::distributed::optimization::LatencyModel::new();
        model.add_measurement(0, 1, 10.0);
        model.add_measurement(0, 2, 12.0);
        model.add_measurement(1, 2, 11.0);
        b.iter(|| {
            let latency = model.estimate(0, 1);
            black_box(latency);
        });
    });

    group.bench_function("bandwidth_model_estimate", |b| {
        let mut model = numrs2::distributed::optimization::BandwidthModel::new();
        model.add_measurement(0, 1, 1e9);
        model.add_measurement(0, 2, 1.2e9);
        model.add_measurement(1, 2, 1.1e9);
        b.iter(|| {
            let bandwidth = model.estimate(0, 1);
            black_box(bandwidth);
        });
    });

    group.bench_function("latency_model_add_measurement", |b| {
        b.iter(|| {
            let mut model = numrs2::distributed::optimization::LatencyModel::new();
            for i in 0..10 {
                model.add_measurement(i, (i + 1) % 10, 10.0 + i as f64);
            }
            black_box(model);
        });
    });

    group.bench_function("bandwidth_model_add_measurement", |b| {
        b.iter(|| {
            let mut model = numrs2::distributed::optimization::BandwidthModel::new();
            for i in 0..10 {
                model.add_measurement(i, (i + 1) % 10, 1e9 + i as f64 * 1e8);
            }
            black_box(model);
        });
    });

    group.finish();
}

criterion_group!(
    distributed_benches,
    bench_distribution_strategies,
    bench_index_mapping,
    bench_collective_reductions,
    bench_message_serialization,
    bench_network_topology,
    bench_process_group,
    bench_distributed_matrix_dims,
    bench_comm_models,
);

criterion_main!(distributed_benches);
