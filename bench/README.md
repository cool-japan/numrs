# NumRS2 Benchmarking Suite

This directory contains benchmarking tools for NumRS2, including performance comparisons with NumPy.

## Overview

The benchmarking suite consists of:

1. **Rust Benchmarks**: Standard Rust benchmarks for NumRS2 operations
2. **Python Benchmarks**: Equivalent NumPy/SciPy benchmarks for comparison
3. **Comparison Utilities**: Tools to compare performance between NumRS2 and NumPy

## Running the Benchmarks

### Step 1: Generate NumPy Reference Data (Optional)

First, run the NumPy benchmarks to generate reference data:

```bash
cd bench/numpy
python benchmark_numpy.py
```

This will create a `numpy_benchmark_results.json` file with NumPy's benchmark results.

### Step 2: Run the NumRS2 Benchmarks

Run the NumRS2 benchmarks using Rust's built-in benchmarking tools:

```bash
# Basic benchmarks
cargo bench

# Benchmarks with SciRS2 integration
cargo bench --features scirs
```

If you've generated NumPy reference data in Step 1, the benchmarks will automatically compare performance with NumPy and print the results.

## Benchmark Categories

The benchmarks cover the following categories:

1. **Array Creation**: Testing the performance of creating arrays with different methods and sizes
2. **Array Operations**: Testing mathematical operations like addition, multiplication, and functions
3. **Linear Algebra**: Testing BLAS/LAPACK operations like matrix multiplication and decompositions
4. **Distributions**: Testing random number generation from various probability distributions

## Interpreting Results

The benchmark results include:

- **Median Time**: The median execution time in nanoseconds (Rust) or milliseconds (NumPy)
- **Comparison**: If NumPy reference data is available, a comparison showing the relative performance

Example output:
```
Benchmark: zeros_medium - NumRS2: 0.05 ms, NumPy: 0.10 ms - NumRS2 is 2.00x faster than NumPy
```

## Adding New Benchmarks

To add new benchmarks:

1. Add a new benchmark function to `bench.rs` following the existing pattern
2. Add an equivalent benchmark to `benchmark_numpy.py` if you want to compare with NumPy
3. Run both benchmarks to compare performance

## Best Practices

For reliable benchmarking:

1. Run benchmarks on the same machine for NumRS2 and NumPy
2. Run benchmarks multiple times to account for system variability
3. Close other applications that might compete for CPU/memory resources
4. Use the same input sizes/parameters for both NumRS2 and NumPy benchmarks

## Analyzing Performance Bottlenecks

If you identify performance issues in NumRS2:

1. Profile the code using tools like `perf` on Linux or `Instruments` on macOS
2. Compare the algorithm complexity between NumRS2 and NumPy
3. Check for memory layout differences that might affect cache efficiency
4. Investigate SIMD usage and parallel processing strategies

## Continuous Integration

The benchmarks can be used in CI to detect performance regressions:

1. Store baseline performance for key operations
2. Run benchmarks on each pull request or commit
3. Alert if performance degrades beyond a threshold

## Benchmark Parameters

You can adjust the following parameters in both benchmark files:

- `SMALL_SIZE`: Size for small array benchmarks (default: 1,000)
- `MEDIUM_SIZE`: Size for medium array benchmarks (default: 10,000)
- `LARGE_SIZE`: Size for large array benchmarks (default: 100,000)
- `ITERATIONS`: Number of iterations for each benchmark (Python only, default: 10)