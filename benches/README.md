# NumRS2 Benchmark Suite

Comprehensive performance benchmarks for NumRS2 v0.2.0 Enhanced features and core functionality.

## Overview

NumRS2's benchmark suite provides detailed performance measurements across all major components:
- Core array operations and linear algebra
- Multi-objective optimization (NSGA-II, NSGA-III)
- Memory-optimized operations
- Parallel algorithm performance
- Expression template system
- FFT and signal processing
- Special mathematical functions

All benchmarks use the [Criterion.rs](https://github.com/bheisler/criterion.rs) framework for statistical rigor and historical tracking.

## Benchmark Suites

### Core Operations

#### `core_operations_benchmark.rs`
Basic array operations including element-wise arithmetic, reductions, and transformations.

#### `linear_algebra_benchmark.rs`
Linear algebra operations: matrix multiplication, decompositions (SVD, QR, Cholesky), eigenvalue computations.

#### `fft_benchmark.rs`
Fast Fourier Transform operations with various sizes and implementations (real/complex FFT).

#### `special_functions_benchmark.rs`
Special mathematical functions: gamma, beta, bessel, error functions, etc.

### v0.2.0 Enhanced Features

#### `multi_objective_benchmark.rs` (NEW)
Multi-objective optimization algorithm performance:
- **NSGA-II**: Population scaling (50, 100, 200), generation counts
- **NSGA-III**: Many-objective optimization (3, 5, 8 objectives)
- **Quality Metrics**: Hypervolume, IGD, GD, Spacing, Spread calculation
- **Test Problems**: ZDT1, ZDT2, ZDT3, DTLZ2, DTLZ3
- **Convergence Analysis**: Per-generation performance, algorithm comparison

#### `memory_optimization_benchmark.rs` (NEW)
Memory-optimized operations vs standard implementations:
- **Reduction Operations**: `sum_optimized` vs `sum` (50-80% faster, zero allocations)
- **Statistical Operations**: `mean_optimized`, `variance_optimized`, `std_optimized`
- **In-place Operations**: `map_inplace` vs `map` (2-3x faster)
- **Buffer Reuse**: `map_to` with pre-allocated buffers (30-50% faster)
- **Batch Operations**: Cumulative allocation reduction benefits
- **SIMD Acceleration**: Threshold analysis (64 elements)

#### `parallel_algorithms_benchmark.rs` (NEW)
Parallel algorithm scaling and efficiency:
- **Operations**: map, reduce, filter, sort, map-reduce, prefix sum
- **Thread Scaling**: 1, 2, 4, 8 threads
- **Strong Scaling**: Fixed problem size, variable threads
- **Weak Scaling**: Problem size scales with threads
- **Work Distribution**: Irregular workload handling
- **Array Sizes**: 10K to 10M elements

### Expression Templates

#### `expression_template_benchmark.rs`
Expression template system performance:
- SIMD-optimized evaluation
- Operation fusion
- Buffer reuse patterns
- Complex expression chains
- Allocation reduction

### Production Benchmarks

#### `production_readiness_benchmark.rs`
Real-world usage patterns and end-to-end workflows.

#### `numpy_comparison_benchmark.rs`
Performance comparison against NumPy operations (when available).

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suite
cargo bench --bench multi_objective_benchmark
cargo bench --bench memory_optimization_benchmark
cargo bench --bench parallel_algorithms_benchmark

# Run specific benchmark within a suite
cargo bench --bench multi_objective_benchmark -- nsga2_zdt1

# Run benchmarks matching a pattern
cargo bench -- "sum_optimized"
```

### Advanced Usage

```bash
# Save baseline for comparison
cargo bench --bench memory_optimization_benchmark -- --save-baseline before_opt

# Compare against baseline
cargo bench --bench memory_optimization_benchmark -- --baseline before_opt

# Generate detailed HTML reports
cargo bench -- --plotting-backend gnuplot

# Profile a specific benchmark
cargo bench --bench parallel_algorithms_benchmark --profile-time=5

# Run with specific sample size
cargo bench -- --sample-size 10
```

### Continuous Integration

```bash
# Quick smoke test (reduced sample size)
cargo bench -- --quick

# Save results for tracking
cargo bench -- --save-baseline ci-$(git rev-parse --short HEAD)
```

## Expected Performance Characteristics

### Multi-Objective Optimization

| Operation | Time Complexity | Target Performance |
|-----------|----------------|-------------------|
| NSGA-II | O(MN²) | ZDT1 100 gen @ 100 pop < 5s |
| NSGA-III | O(MN log N) | DTLZ2 50 gen @ 100 pop < 8s |
| Hypervolume | O(N^(M-1)) | 100 points, 2 obj < 10ms |
| IGD/GD | O(N·R) | 100 points < 5ms |
| Spacing | O(N²) | 200 points < 20ms |

Where:
- M = number of objectives
- N = population size
- R = reference front size

### Memory Optimization

| Operation | Speedup | Allocation Reduction |
|-----------|---------|---------------------|
| `sum_optimized` | 1.5-2x | 100% (zero alloc) |
| `mean_optimized` | 1.5-2x | 100% (zero alloc) |
| `map_inplace` | 2-3x | 100% (zero alloc) |
| `map_to` | 1.3-1.5x | 100% (reuse buffer) |
| Batch ops (10x) | 2-4x | 90% cumulative |

**SIMD Threshold**: 64 elements
- Below: Scalar fallback
- Above: SIMD acceleration (2-4x faster)

### Parallel Algorithms

| Array Size | Target Efficiency (4 threads) | Speedup |
|------------|------------------------------|---------|
| 10K | > 70% | > 2.8x |
| 100K | > 85% | > 3.4x |
| 1M | > 90% | > 3.6x |
| 10M | > 92% | > 3.7x |

**Efficiency** = Speedup / Thread Count

**Parallel Threshold**: 1,000 elements
- Below: Sequential execution
- Above: Parallel execution

### Expression Templates

| Pattern | Benefit | Performance |
|---------|---------|-------------|
| SIMD evaluation | 2-4x faster | 100K elements < 1ms |
| Operation fusion | Reduced allocations | 2x faster for chains |
| Buffer reuse | No allocation | 3x faster for loops |

## Interpreting Results

### Criterion Output

```
sum_optimized/100        time:   [245.67 ns 248.32 ns 251.48 ns]
                        thrpt:  [397.67 Melem/s 402.71 Melem/s 407.21 Melem/s]
                 change: [-5.2341% -3.8923% -2.4156%] (p = 0.00 < 0.05)
                        Performance has improved.
```

**Key Metrics**:
- **time**: Mean execution time with confidence interval
- **thrpt**: Throughput (elements/second)
- **change**: Performance change from previous run
- **p-value**: Statistical significance (< 0.05 = significant)

### Performance Targets

✅ **Good**: Within 10% of target
⚠️ **Acceptable**: Within 20% of target
❌ **Regression**: > 20% slower than target or previous baseline

### Statistical Significance

- **p < 0.05**: Statistically significant change
- **Confidence Interval**: Narrower is better (more consistent)
- **Outliers**: Check for thermal throttling or background processes

## Profiling and Optimization

### Flamegraph Profiling

```bash
# Install flamegraph
cargo install flamegraph

# Profile specific benchmark
cargo flamegraph --bench multi_objective_benchmark -- --bench

# View flamegraph.svg in browser
```

### Linux perf

```bash
# Record performance data
perf record --call-graph=dwarf cargo bench --bench parallel_algorithms_benchmark

# View report
perf report

# Annotate assembly
perf annotate
```

### Memory Profiling

```bash
# Valgrind massif (heap profiling)
valgrind --tool=massif cargo bench --bench memory_optimization_benchmark -- --profile-time=5

# View results
ms_print massif.out.*

# DHAT (dynamic heap analysis)
valgrind --tool=dhat cargo bench --bench memory_optimization_benchmark -- --profile-time=5
```

### Criterion Built-in Profiling

```bash
# Profile with sampling profiler
cargo bench --bench multi_objective_benchmark -- --profile-time=5

# Results in target/criterion/<benchmark>/profile/
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Benchmarks

on:
  pull_request:
  push:
    branches: [main, master]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run benchmarks
        run: |
          cargo bench --bench memory_optimization_benchmark -- --save-baseline pr-${{ github.event.pull_request.number }}

      - name: Compare to master
        run: |
          git fetch origin master
          git checkout master
          cargo bench --bench memory_optimization_benchmark -- --save-baseline master
          git checkout -
          cargo bench --bench memory_optimization_benchmark -- --baseline master
```

### Regression Detection

**Automated Thresholds**:
- Core operations: > 10% regression
- Optimization features: > 15% regression
- Complex algorithms: > 20% regression

**Manual Review Required**:
- New features without baselines
- Algorithm changes
- Platform-specific behavior

## Contributing

### Adding New Benchmarks

1. **Create benchmark file**: `benches/my_feature_benchmark.rs`

2. **Follow structure**:
```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_my_feature(c: &mut Criterion) {
    c.bench_function("my_feature", |b| {
        b.iter(|| {
            // Benchmark code here
        });
    });
}

criterion_group!(benches, bench_my_feature);
criterion_main!(benches);
```

3. **Add to Cargo.toml**:
```toml
[[bench]]
name = "my_feature_benchmark"
path = "benches/my_feature_benchmark.rs"
harness = false
```

### Benchmark Naming Conventions

- **Function names**: `bench_<feature>_<aspect>` (e.g., `bench_sum_optimized`)
- **Group names**: `<feature>_<aspect>` (e.g., `parallel_map_scaling`)
- **Benchmark IDs**: `<variant>_<param>` (e.g., `threads_4t_1M`)

### Required Configuration

```rust
// Set throughput for meaningful comparison
group.throughput(Throughput::Elements(size as u64));

// Reduce sample size for expensive operations
group.sample_size(10);

// Set measurement time for fast operations
group.measurement_time(Duration::from_secs(1));
```

## Performance Tracking

### Historical Data

Criterion automatically stores historical data in `target/criterion/`.

```bash
# View history for specific benchmark
criterion-view target/criterion/sum_optimized/
```

### Comparison Tools

```bash
# critcmp (criterion comparison tool)
cargo install critcmp

# Compare baselines
critcmp before_opt after_opt

# Generate comparison table
critcmp --export before_opt after_opt > comparison.md
```

### Performance Dashboard

Consider using tools like:
- [Bencher](https://bencher.dev/) - Continuous benchmarking platform
- [Criterion Dashboard](https://github.com/bheisler/criterion.rs/blob/master/book/src/user_guide/html_reports.md) - Built-in HTML reports

## Troubleshooting

### Inconsistent Results

**Causes**:
- Thermal throttling
- Background processes
- Power management

**Solutions**:
```bash
# Disable CPU frequency scaling (Linux)
sudo cpupower frequency-set --governor performance

# Pin to specific cores
taskset -c 0-3 cargo bench

# Increase sample size
cargo bench -- --sample-size 100
```

### Build Issues

```bash
# Clean rebuild
cargo clean
cargo bench

# Check dependencies
cargo tree | grep criterion

# Verbose output
cargo bench --verbose
```

### Memory Issues

```bash
# Increase stack size
RUST_MIN_STACK=8388608 cargo bench

# Check for leaks
valgrind --leak-check=full cargo bench --bench memory_optimization_benchmark
```

## Resources

- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/)
- [NumRS2 Documentation](https://docs.rs/numrs2)
- [Performance Optimization Guide](../docs/PERFORMANCE.md)
- [SCIRS2 Integration Policy](../SCIRS2_INTEGRATION_POLICY.md)

## License

Apache-2.0 - Copyright (c) COOLJAPAN OU (Team Kitasan)
