# NumRS2 Benchmarking Guide

Comprehensive guide for running, interpreting, and using benchmarks in NumRS2 v1.0.0 (0.2.0 release).

## Table of Contents

1. [Overview](#overview)
2. [Benchmark Suite](#benchmark-suite)
3. [Running Benchmarks](#running-benchmarks)
4. [Interpreting Results](#interpreting-results)
5. [Performance Optimization Tips](#performance-optimization-tips)
6. [Hardware Requirements](#hardware-requirements)
7. [Comparison with NumPy](#comparison-with-numpy)
8. [Troubleshooting](#troubleshooting)

## Overview

NumRS2 includes a comprehensive benchmark suite built with the [Criterion.rs](https://github.com/bheisler/criterion.rs) library. The benchmarks cover all major operations and are designed to:

- Track performance across releases
- Identify performance regressions
- Compare SIMD vs scalar performance
- Evaluate parallel processing efficiency
- Measure memory bandwidth utilization
- Compare with NumPy where applicable

## Benchmark Suite

### 1. Linear Algebra Benchmarks (`linalg_benchmarks`)

**File:** `bench/linalg_benchmarks.rs`

**Operations tested:**
- Matrix multiplication (10x10 to 1000x1000)
- Matrix-vector multiplication
- Matrix transpose (square and rectangular)
- Matrix inverse (10x10 to 200x200)
- Determinant calculation
- Matrix norms (Frobenius, infinity, 1-norm)
- QR decomposition
- Cholesky decomposition
- SVD (Singular Value Decomposition)
- LU decomposition
- Eigenvalue decomposition
- Linear system solving
- Matrix rank and condition number
- Matrix trace
- Outer and inner products
- Cross product (3D)
- Kronecker product

**Run:**
```bash
cargo bench --bench linalg_benchmarks
```

### 2. Statistics Benchmarks (`stats_benchmarks`)

**File:** `bench/stats_benchmarks.rs`

**Operations tested:**
- Basic statistics (mean, variance, std, median)
- Quantiles and percentiles
- Correlation and covariance (vectors and matrices)
- Histogram computation (10, 50, 100 bins)
- Distribution sampling:
  - Normal (standard and custom)
  - Uniform
  - Exponential
  - Gamma
  - Beta
  - Chi-squared
  - Student's t
  - F distribution
  - Poisson
  - Binomial
- Cumulative statistics (cumsum, cumprod)
- Statistical moments (skewness, kurtosis)
- Random sampling and shuffling

**Run:**
```bash
cargo bench --bench stats_benchmarks
```

### 3. FFT Benchmarks (`fft_benchmarks`)

**File:** `bench/fft_benchmarks.rs`

**Operations tested:**
- 1D FFT/IFFT (64 to 16384 points)
- Real FFT/IRFFT
- 2D FFT/IFFT (8x8 to 256x256)
- 2D Real FFT/IRFFT
- Window functions (rectangular, Hann, Hamming, Blackman)
- FFT shift operations
- Frequency axis generation
- Power spectrum calculation
- FFT on different signal types (random, sine, square, impulse)
- End-to-end FFT workflow
- Implementation comparison
- Data type comparison (f32 vs f64)

**Run:**
```bash
cargo bench --bench fft_benchmarks
```

### 4. Array Operations Benchmarks (`array_ops_benchmarks`)

**File:** `bench/array_ops_benchmarks.rs`

**Operations tested:**
- Element-wise operations (add, sub, mul, div, pow)
- Broadcasting (scalar to array, vector to matrix)
- Reduction operations (sum, prod, min, max, argmin, argmax)
- Array indexing (element access)
- Array slicing (1D and 2D)
- Array reshaping (1D to 2D, flattening)
- Array transposition (square and rectangular)
- Array concatenation (1D and 2D, different axes)
- Array stacking (vstack, hstack)
- Array splitting
- Array tiling and repetition

**Run:**
```bash
cargo bench --bench array_ops_benchmarks
```

### 5. Optimization Benchmarks (`optimization_benchmarks`)

**File:** `bench/optimization_benchmarks.rs`

**Operations tested:**
- BFGS optimization (2D to 10D)
- L-BFGS optimization (2D to 20D)
- Conjugate gradient methods:
  - Fletcher-Reeves
  - Polak-Ribiere
  - Hestenes-Stiefel
- Trust region methods
- Genetic algorithms
- Particle swarm optimization
- Simulated annealing
- Differential evolution
- Algorithm comparison

**Test functions:**
- Rosenbrock
- Sphere
- Rastrigin
- Ackley

**Run:**
```bash
cargo bench --bench optimization_benchmarks
```

### 6. SIMD Comparison Benchmarks (`simd_comparison_benchmark`)

**File:** `bench/simd_comparison_benchmark.rs`

**Operations tested:**
- SIMD vs scalar addition
- SIMD vs scalar multiplication
- SIMD vs scalar dot product
- SIMD vs scalar sum reduction
- Threshold analysis (2 to 256 elements)
- Data type comparison (f32 vs f64)
- Alignment effects
- Complex operations (FMA, norm)
- Strided data access
- Memory bandwidth with SIMD

**Run:**
```bash
cargo bench --bench simd_comparison_benchmark
```

### 7. Parallel Benchmarks (`parallel_benchmarks`)

**File:** `bench/parallel_benchmarks.rs`

**Operations tested:**
- Parallel vs sequential sum
- Parallel vs sequential matrix multiplication
- Parallel reduction operations
- Parallel map operations
- Thread scaling analysis
- Parallel overhead for different array sizes
- Load balancing efficiency
- Parallel matrix operations
- Parallel statistics
- Parallel FFT

**Run:**
```bash
cargo bench --bench parallel_benchmarks
```

### 8. Memory Benchmarks (`memory_benchmarks`)

**File:** `bench/memory_benchmarks.rs`

**Operations tested:**
- Memory allocation patterns (1D, 2D, zeros, ones)
- Cache efficiency (row-major vs column-major)
- Memory bandwidth utilization (read, write, copy, triad)
- Copy vs view operations
- In-place vs allocating operations
- Memory access patterns (sequential, strided, random)
- Cache line effects
- Allocation size effects (small, medium, large)
- Contiguous vs non-contiguous memory
- Prefetching effects

**Run:**
```bash
cargo bench --bench memory_benchmarks
```

## Running Benchmarks

### Run All Benchmarks

```bash
cargo bench
```

### Run Specific Benchmark Suite

```bash
cargo bench --bench linalg_benchmarks
cargo bench --bench stats_benchmarks
cargo bench --bench fft_benchmarks
cargo bench --bench array_ops_benchmarks
cargo bench --bench optimization_benchmarks
cargo bench --bench simd_comparison_benchmark
cargo bench --bench parallel_benchmarks
cargo bench --bench memory_benchmarks
```

### Run Specific Benchmark Function

```bash
# Run only matrix multiplication benchmarks
cargo bench --bench linalg_benchmarks -- matrix_multiplication

# Run only FFT 1D benchmarks
cargo bench --bench fft_benchmarks -- fft_1d

# Run only SIMD threshold analysis
cargo bench --bench simd_comparison_benchmark -- threshold_analysis
```

### Save Results for Comparison

```bash
# Save baseline
cargo bench -- --save-baseline main

# Make changes...

# Compare with baseline
cargo bench -- --baseline main
```

### Generate HTML Reports

Criterion automatically generates HTML reports in `target/criterion/`. Open them with:

```bash
# macOS
open target/criterion/report/index.html

# Linux
xdg-open target/criterion/report/index.html

# Windows
start target/criterion/report/index.html
```

## Interpreting Results

### Understanding Criterion Output

```
matrix_multiplication/square_matmul/100
                        time:   [1.2345 ms 1.2567 ms 1.2789 ms]
                        change: [-5.2341% -3.1234% -1.0123%] (p = 0.02 < 0.05)
                        Performance has improved.
```

**Components:**
- **time**: Median measurement with 95% confidence interval
  - Lower bound: 1.2345 ms
  - Median: 1.2567 ms
  - Upper bound: 1.2789 ms
- **change**: Relative change from previous run
  - Negative = faster (improvement)
  - Positive = slower (regression)
- **p-value**: Statistical significance (< 0.05 = significant)

### Performance Metrics

1. **Throughput**: Operations per second
   - Higher is better
   - Compare with theoretical peak performance

2. **Latency**: Time per operation
   - Lower is better
   - Important for real-time applications

3. **Scaling**: Performance vs problem size
   - O(n), O(n²), O(n³) complexity
   - SIMD speedup: 2-4x for f64, 4-8x for f32
   - Parallel speedup: Near-linear with thread count

4. **Efficiency**: Actual vs theoretical performance
   - Memory bandwidth utilization
   - Cache hit rates
   - SIMD utilization

## Performance Optimization Tips

### 1. Choose Appropriate Data Types

- **f32 vs f64**: Use f32 when precision allows (2x SIMD lanes)
- **Integer types**: Use smallest type that fits your range

### 2. Memory Layout

- **Contiguous arrays**: Fastest access pattern
- **Row-major order**: Default in NumRS2 (same as NumPy)
- **Avoid unnecessary transposes**: Cache-unfriendly

### 3. SIMD Optimization

- **Minimum size**: SIMD benefits start at ~64 elements
- **Alignment**: Aligned data is faster (handled automatically)
- **Contiguous data**: SIMD requires contiguous memory

### 4. Parallel Processing

- **Minimum size**: Parallel benefits start at ~10,000 elements
- **Thread count**: Optimal = number of physical cores
- **Overhead**: Consider serial for small arrays

### 5. Cache Optimization

- **Locality**: Access nearby elements together
- **Blocking**: Use cache-sized blocks for large matrices
- **Prefetching**: Sequential access enables hardware prefetch

### 6. Algorithm Selection

- **Matrix multiplication**: O(n³) - consider size limits
- **SVD**: Expensive - use only when needed
- **Iterative solvers**: Better for large sparse systems

## Hardware Requirements

### Minimum Requirements

- **CPU**: x86_64 or ARM64 with SIMD support
- **RAM**: 4 GB (8 GB recommended)
- **Disk**: 1 GB for build artifacts

### Recommended Hardware

- **CPU**: Modern multi-core processor (4+ cores)
  - x86_64: AVX2 or AVX-512 support
  - ARM64: NEON support
- **RAM**: 16 GB or more
- **Disk**: SSD for faster compilation

### Performance Expectations

**CPU-bound operations:**
- Matrix multiplication: ~100 GFLOPS on modern CPUs
- FFT: ~1-10 GB/s throughput
- Element-wise operations: Memory bandwidth limited

**Memory-bound operations:**
- Stream bandwidth: 10-100 GB/s (DDR4)
- L1 cache: ~1 TB/s
- L2 cache: ~200 GB/s
- L3 cache: ~100 GB/s

## Comparison with NumPy

### NumRS2 Advantages

1. **Zero-copy operations**: Views don't allocate
2. **Pure Rust**: No C/Fortran dependencies
3. **Type safety**: Compile-time error checking
4. **Memory safety**: No segfaults or undefined behavior

### Performance Comparison

**Expected relative performance:**

| Operation | NumRS2 vs NumPy |
|-----------|-----------------|
| Matrix multiplication (small) | 0.8-1.2x |
| Matrix multiplication (large) | 0.9-1.1x |
| Element-wise operations | 0.9-1.2x |
| FFT | 0.8-1.0x |
| Statistics | 1.0-1.5x |
| Memory allocation | 1.0-1.3x |

**Notes:**
- NumPy uses MKL/OpenBLAS (highly optimized C/Fortran)
- NumRS2 uses OxiBLAS (pure Rust, actively improving)
- Performance varies by operation and hardware

### Running Comparison Benchmarks

```bash
# NumRS2 benchmarks
cargo bench

# NumPy benchmarks (requires Python setup)
cd bench
python numpy_benchmark.py
```

## Troubleshooting

### Issue: Benchmarks Take Too Long

**Solution 1**: Run subset of benchmarks
```bash
cargo bench --bench linalg_benchmarks -- matrix_multiplication
```

**Solution 2**: Reduce sample size (in benchmark code)
```rust
group.sample_size(10);  // Default is 100
```

**Solution 3**: Use quick benchmark mode
```bash
cargo bench -- --quick
```

### Issue: Inconsistent Results

**Possible causes:**
- System load (close other applications)
- CPU frequency scaling (disable for benchmarking)
- Thermal throttling (ensure adequate cooling)
- Background processes (disable antivirus, etc.)

**Solutions:**
```bash
# Linux: Disable CPU frequency scaling
sudo cpupower frequency-set --governor performance

# Check system load
top
htop
```

### Issue: Out of Memory

**Solution 1**: Run smaller benchmarks
```bash
cargo bench --bench memory_benchmarks -- small
```

**Solution 2**: Increase swap space

**Solution 3**: Skip large problem sizes
- Edit benchmark files to reduce maximum sizes

### Issue: Compilation Errors

**Current known issue**: There are compilation errors in `src/optimize/simulated_annealing.rs` related to `NumRs2Error::Other` variant not existing. These need to be fixed before benchmarks can run.

**Solution**: Fix the error enum issues first:
```bash
# Check error definition
cat src/error/legacy.rs

# Fix simulated_annealing.rs to use correct error variant
```

### Issue: Performance Lower Than Expected

**Check:**
1. **Build mode**: Ensure using `--release` or `cargo bench`
2. **CPU frequency**: Check for thermal throttling
3. **SIMD support**: Verify CPU features
4. **Thread count**: Check `RAYON_NUM_THREADS`
5. **Memory**: Ensure no swapping

**Verify:**
```bash
# Check if release mode
cargo bench --verbose

# Check CPU features
lscpu | grep Flags  # Linux
sysctl -a | grep cpu  # macOS

# Check memory usage
free -h  # Linux
vm_stat  # macOS
```

## Best Practices

### Before Benchmarking

1. **Close unnecessary applications**
2. **Disable CPU frequency scaling**
3. **Ensure adequate cooling**
4. **Use AC power (laptops)**
5. **Wait for system to stabilize**

### During Benchmarking

1. **Don't use the system**
2. **Monitor temperature**
3. **Save baselines regularly**
4. **Document system configuration**

### After Benchmarking

1. **Archive results**
2. **Compare with previous runs**
3. **Generate reports**
4. **Document findings**

## Performance Regression Testing

### Automated Testing

```bash
# Save baseline before changes
cargo bench -- --save-baseline before

# Make changes...

# Compare with baseline
cargo bench -- --baseline before

# Check for regressions (exit code != 0 if regression)
cargo bench -- --baseline before || echo "Performance regression detected!"
```

### CI/CD Integration

Example GitHub Actions workflow:
```yaml
name: Benchmark

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Run benchmarks
        run: cargo bench --no-fail-fast
      - name: Archive results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: target/criterion/
```

## Additional Resources

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [NumRS2 Documentation](https://docs.rs/numrs2)
- [SciRS2 Performance Guide](https://github.com/cool-japan/scirs2/docs/PERFORMANCE.md)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)

## Contributing

To add new benchmarks:

1. Create benchmark file in `bench/`
2. Add entry in `Cargo.toml`
3. Follow existing patterns
4. Document in this guide
5. Test thoroughly
6. Submit pull request

## License

NumRS2 is licensed under Apache-2.0.

Copyright © 2025 COOLJAPAN OU (Team KitaSan)
