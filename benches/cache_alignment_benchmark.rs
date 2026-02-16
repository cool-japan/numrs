//! Cache Alignment Performance Benchmark
//!
//! Comprehensive benchmarks measuring the actual performance impact of cache alignment
//! optimizations (`#[repr(align(64))]`) on:
//! 1. False sharing elimination in parallel workloads
//! 2. SIMD load/store performance (aligned vs unaligned)
//! 3. Parallel configuration access patterns
//! 4. Cache line contention with thread scaling
//!
//! These benchmarks complement the memory_optimization_benchmark by specifically
//! isolating cache alignment effects rather than allocation reduction.
//!
//! Reference: /tmp/CACHE_ALIGNMENT_PERFORMANCE_IMPACT.md

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use numrs2::parallel_optimize::{ParallelConfig, SchedulingStrategy};
use std::hint::black_box;
use std::sync::Arc;
use std::thread;

// Import SIMD operations from SciRS2 ecosystem
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::simd::reductions::{simd_max_f64, simd_min_f64, simd_sum_f64};

/// Test data size for SIMD benchmarks (must be divisible by 8 for AVX operations)
const SIMD_TEST_SIZE: usize = 1024;

/// Thread counts to test for parallel benchmarks
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8];

/// Iteration count for contention tests
const CONTENTION_ITERATIONS: usize = 10_000;

// ============================================================================
// TEST STRUCTURES: Aligned vs Unaligned
// ============================================================================

/// Aligned counter structure (occupies full cache line, prevents false sharing)
#[repr(align(64))]
struct AlignedCounter {
    value: std::sync::atomic::AtomicU64,
    _pad: [u8; 56], // Pad to 64 bytes (8 + 56 = 64)
}

impl Default for AlignedCounter {
    fn default() -> Self {
        Self {
            value: std::sync::atomic::AtomicU64::new(0),
            _pad: [0; 56],
        }
    }
}

/// Unaligned counter structure (may share cache line, causes false sharing)
#[derive(Default)]
struct UnalignedCounter {
    value: std::sync::atomic::AtomicU64,
}

/// Aligned data array for SIMD operations
#[repr(align(64))]
struct AlignedArray {
    data: [f64; SIMD_TEST_SIZE],
}

impl AlignedArray {
    fn new() -> Self {
        Self {
            data: [0.0; SIMD_TEST_SIZE],
        }
    }

    fn init_with_pattern(&mut self) {
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = (i as f64) * 0.5;
        }
    }
}

/// Unaligned data array for SIMD operations
struct UnalignedArray {
    data: [f64; SIMD_TEST_SIZE],
}

impl UnalignedArray {
    fn new() -> Self {
        Self {
            data: [0.0; SIMD_TEST_SIZE],
        }
    }

    fn init_with_pattern(&mut self) {
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = (i as f64) * 0.5;
        }
    }
}

// ============================================================================
// BENCHMARK 1: False Sharing in Parallel Counters
// ============================================================================

/// Benchmark parallel access to ALIGNED counters (no false sharing expected)
///
/// Expected: Linear scaling with thread count, minimal cache coherency overhead
fn bench_false_sharing_aligned(c: &mut Criterion) {
    let mut group = c.benchmark_group("false_sharing/aligned");

    for &num_threads in THREAD_COUNTS.iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |bencher, &num_threads| {
                bencher.iter(|| {
                    // Create array of aligned counters
                    let counters: Vec<AlignedCounter> = (0..num_threads)
                        .map(|_| AlignedCounter::default())
                        .collect();
                    let counters = Arc::new(counters);

                    // Spawn threads, each incrementing its own counter
                    let handles: Vec<_> = (0..num_threads)
                        .map(|thread_id| {
                            let counters = Arc::clone(&counters);
                            thread::spawn(move || {
                                for _ in 0..CONTENTION_ITERATIONS {
                                    counters[thread_id]
                                        .value
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                            })
                        })
                        .collect();

                    // Wait for all threads
                    for handle in handles {
                        handle.join().unwrap();
                    }

                    // Verify result
                    let sum: u64 = counters
                        .iter()
                        .map(|c| c.value.load(std::sync::atomic::Ordering::Relaxed))
                        .sum();
                    black_box(sum);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark parallel access to UNALIGNED counters (false sharing expected)
///
/// Expected: Poor scaling with thread count, high cache coherency overhead
fn bench_false_sharing_unaligned(c: &mut Criterion) {
    let mut group = c.benchmark_group("false_sharing/unaligned");

    for &num_threads in THREAD_COUNTS.iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |bencher, &num_threads| {
                bencher.iter(|| {
                    // Create array of unaligned counters (packed tightly, causing false sharing)
                    let counters: Vec<UnalignedCounter> = (0..num_threads)
                        .map(|_| UnalignedCounter::default())
                        .collect();
                    let counters = Arc::new(counters);

                    // Spawn threads, each incrementing its own counter
                    let handles: Vec<_> = (0..num_threads)
                        .map(|thread_id| {
                            let counters = Arc::clone(&counters);
                            thread::spawn(move || {
                                for _ in 0..CONTENTION_ITERATIONS {
                                    counters[thread_id]
                                        .value
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                            })
                        })
                        .collect();

                    // Wait for all threads
                    for handle in handles {
                        handle.join().unwrap();
                    }

                    // Verify result
                    let sum: u64 = counters
                        .iter()
                        .map(|c| c.value.load(std::sync::atomic::Ordering::Relaxed))
                        .sum();
                    black_box(sum);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 2: SIMD Loads (Aligned vs Unaligned)
// ============================================================================

/// Benchmark SIMD operations on ALIGNED arrays
///
/// Expected: Optimal SIMD performance, single instruction per 64-byte load
fn bench_simd_aligned_loads(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_loads/aligned");
    group.throughput(Throughput::Elements(SIMD_TEST_SIZE as u64));

    let mut array = AlignedArray::new();
    array.init_with_pattern();

    group.bench_function("simd_sum", |bencher| {
        bencher.iter(|| {
            // Use SciRS2 SIMD operations (follows SCIRS2 POLICY)
            let view = ArrayView1::from(&array.data);
            let sum: f64 = simd_sum_f64(&view);
            black_box(sum);
        });
    });

    group.bench_function("simd_min", |bencher| {
        bencher.iter(|| {
            let view = ArrayView1::from(&array.data);
            let min: f64 = simd_min_f64(&view);
            black_box(min);
        });
    });

    group.bench_function("simd_max", |bencher| {
        bencher.iter(|| {
            let view = ArrayView1::from(&array.data);
            let max: f64 = simd_max_f64(&view);
            black_box(max);
        });
    });

    group.finish();
}

/// Benchmark SIMD operations on UNALIGNED arrays
///
/// Expected: Degraded SIMD performance, potential multiple micro-ops per load
fn bench_simd_unaligned_loads(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_loads/unaligned");
    group.throughput(Throughput::Elements(SIMD_TEST_SIZE as u64));

    let mut array = UnalignedArray::new();
    array.init_with_pattern();

    group.bench_function("simd_sum", |bencher| {
        bencher.iter(|| {
            let view = ArrayView1::from(&array.data);
            let sum: f64 = simd_sum_f64(&view);
            black_box(sum);
        });
    });

    group.bench_function("simd_min", |bencher| {
        bencher.iter(|| {
            let view = ArrayView1::from(&array.data);
            let min: f64 = simd_min_f64(&view);
            black_box(min);
        });
    });

    group.bench_function("simd_max", |bencher| {
        bencher.iter(|| {
            let view = ArrayView1::from(&array.data);
            let max: f64 = simd_max_f64(&view);
            black_box(max);
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 3: ParallelConfig Contention
// ============================================================================

/// Benchmark multiple threads reading ALIGNED ParallelConfig
///
/// Expected: Minimal contention, each thread caches config independently
fn bench_parallel_config_aligned(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_config/aligned");

    // ParallelConfig is already aligned to 64 bytes in the source code
    let config = Arc::new(ParallelConfig {
        use_parallel: true,
        min_parallel_size: 1000,
        chunk_size: 250,
        max_threads: Some(8),
        scheduling_strategy: SchedulingStrategy::Adaptive,
    });

    for &num_threads in THREAD_COUNTS.iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |bencher, &num_threads| {
                bencher.iter(|| {
                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let config = Arc::clone(&config);
                            thread::spawn(move || {
                                let mut sum = 0usize;
                                for _ in 0..CONTENTION_ITERATIONS {
                                    // Read config fields (simulating typical access pattern)
                                    sum += config.min_parallel_size;
                                    sum += config.chunk_size;
                                    if config.use_parallel {
                                        sum += 1;
                                    }
                                }
                                black_box(sum);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark to demonstrate what would happen with UNALIGNED config
///
/// Note: We can't directly create an unaligned ParallelConfig since it's
/// already aligned in the source. This benchmark uses an unaligned struct
/// with the same size to simulate the effect.
fn bench_parallel_config_unaligned(c: &mut Criterion) {
    #[derive(Clone)]
    struct UnalignedConfig {
        use_parallel: bool,
        min_parallel_size: usize,
        chunk_size: usize,
        max_threads: Option<usize>,
        _extra: u32, // Make it similar size
    }

    let mut group = c.benchmark_group("parallel_config/unaligned");

    let config = Arc::new(UnalignedConfig {
        use_parallel: true,
        min_parallel_size: 1000,
        chunk_size: 250,
        max_threads: Some(8),
        _extra: 0,
    });

    for &num_threads in THREAD_COUNTS.iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |bencher, &num_threads| {
                bencher.iter(|| {
                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let config = Arc::clone(&config);
                            thread::spawn(move || {
                                let mut sum = 0usize;
                                for _ in 0..CONTENTION_ITERATIONS {
                                    sum += config.min_parallel_size;
                                    sum += config.chunk_size;
                                    if config.use_parallel {
                                        sum += 1;
                                    }
                                }
                                black_box(sum);
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 4: Cache Line Contention with Mixed Access
// ============================================================================

/// Benchmark mixed read/write access to aligned shared state
fn bench_cache_contention_aligned(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_contention/aligned");

    #[repr(align(64))]
    struct AlignedSharedState {
        counter: std::sync::atomic::AtomicU64,
        _pad: [u8; 56],
    }

    for &num_threads in THREAD_COUNTS.iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |bencher, &num_threads| {
                bencher.iter(|| {
                    let state = Arc::new(AlignedSharedState {
                        counter: std::sync::atomic::AtomicU64::new(0),
                        _pad: [0; 56],
                    });

                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let state = Arc::clone(&state);
                            thread::spawn(move || {
                                for _ in 0..CONTENTION_ITERATIONS {
                                    // Read
                                    let val =
                                        state.counter.load(std::sync::atomic::Ordering::Acquire);
                                    // Write
                                    state
                                        .counter
                                        .store(val + 1, std::sync::atomic::Ordering::Release);
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let final_val = state.counter.load(std::sync::atomic::Ordering::Relaxed);
                    black_box(final_val);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mixed read/write access to unaligned shared state
fn bench_cache_contention_unaligned(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_contention/unaligned");

    struct UnalignedSharedState {
        counter: std::sync::atomic::AtomicU64,
    }

    for &num_threads in THREAD_COUNTS.iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_threads),
            &num_threads,
            |bencher, &num_threads| {
                bencher.iter(|| {
                    let state = Arc::new(UnalignedSharedState {
                        counter: std::sync::atomic::AtomicU64::new(0),
                    });

                    let handles: Vec<_> = (0..num_threads)
                        .map(|_| {
                            let state = Arc::clone(&state);
                            thread::spawn(move || {
                                for _ in 0..CONTENTION_ITERATIONS {
                                    // Read
                                    let val =
                                        state.counter.load(std::sync::atomic::Ordering::Acquire);
                                    // Write
                                    state
                                        .counter
                                        .store(val + 1, std::sync::atomic::Ordering::Release);
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }

                    let final_val = state.counter.load(std::sync::atomic::Ordering::Relaxed);
                    black_box(final_val);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 5: Array Stride Access (Cache Miss Patterns)
// ============================================================================

/// Benchmark sequential access to aligned arrays (cache-friendly)
fn bench_stride_aligned_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("stride_access/aligned_sequential");
    group.throughput(Throughput::Elements(SIMD_TEST_SIZE as u64));

    let mut array = AlignedArray::new();
    array.init_with_pattern();

    group.bench_function("sum", |bencher| {
        bencher.iter(|| {
            let sum: f64 = array.data.iter().sum();
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark strided access to aligned arrays (potential cache misses)
fn bench_stride_aligned_strided(c: &mut Criterion) {
    let mut group = c.benchmark_group("stride_access/aligned_strided");
    group.throughput(Throughput::Elements((SIMD_TEST_SIZE / 8) as u64));

    let mut array = AlignedArray::new();
    array.init_with_pattern();

    group.bench_function("sum_stride_8", |bencher| {
        bencher.iter(|| {
            let sum: f64 = array.data.iter().step_by(8).sum();
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark sequential access to unaligned arrays
fn bench_stride_unaligned_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("stride_access/unaligned_sequential");
    group.throughput(Throughput::Elements(SIMD_TEST_SIZE as u64));

    let mut array = UnalignedArray::new();
    array.init_with_pattern();

    group.bench_function("sum", |bencher| {
        bencher.iter(|| {
            let sum: f64 = array.data.iter().sum();
            black_box(sum);
        });
    });

    group.finish();
}

/// Benchmark strided access to unaligned arrays
fn bench_stride_unaligned_strided(c: &mut Criterion) {
    let mut group = c.benchmark_group("stride_access/unaligned_strided");
    group.throughput(Throughput::Elements((SIMD_TEST_SIZE / 8) as u64));

    let mut array = UnalignedArray::new();
    array.init_with_pattern();

    group.bench_function("sum_stride_8", |bencher| {
        bencher.iter(|| {
            let sum: f64 = array.data.iter().step_by(8).sum();
            black_box(sum);
        });
    });

    group.finish();
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group!(
    cache_benches,
    // False sharing benchmarks
    bench_false_sharing_aligned,
    bench_false_sharing_unaligned,
    // SIMD alignment benchmarks
    bench_simd_aligned_loads,
    bench_simd_unaligned_loads,
    // ParallelConfig contention benchmarks
    bench_parallel_config_aligned,
    bench_parallel_config_unaligned,
    // Cache line contention benchmarks
    bench_cache_contention_aligned,
    bench_cache_contention_unaligned,
    // Stride access pattern benchmarks
    bench_stride_aligned_sequential,
    bench_stride_aligned_strided,
    bench_stride_unaligned_sequential,
    bench_stride_unaligned_strided,
);

criterion_main!(cache_benches);
