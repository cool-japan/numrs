# SIMD vs Scalar Performance Benchmarks

## Overview

The `simd_vs_scalar_benchmark.rs` provides comprehensive performance comparisons between SIMD-optimized and scalar implementations of NumRS2 operations. This benchmark suite helps:

1. **Validate SIMD Optimizations**: Demonstrate measurable performance gains from vectorization
2. **Identify Bottlenecks**: Find operations that could benefit from further optimization
3. **Guide Development**: Prioritize SIMD implementation efforts based on performance impact
4. **Document Performance**: Provide concrete numbers for performance claims

## Benchmark Categories

### Mathematical Functions
- **exp**: Exponential function (e^x)
- **log**: Natural logarithm (ln(x))
- **sin**: Sine function
- **cos**: Cosine function
- **sqrt**: Square root
- **abs**: Absolute value

### Reduction Operations
- **sum**: Sum all elements
- **mean**: Calculate mean/average

### Element-wise Operations
- **add**: Element-wise addition
- **multiply**: Element-wise multiplication

## Array Sizes

Benchmarks test four different array sizes to capture various performance characteristics:

- **SMALL (64 elements)**: Below SIMD threshold, expect minimal speedup
- **MEDIUM (512 elements)**: SIMD beneficial, good speedup expected
- **LARGE (4096 elements)**: Maximum SIMD benefit
- **HUGE (32768 elements)**: Tests cache effects and memory bandwidth

## Running the Benchmarks

### Run All SIMD Benchmarks
```bash
cargo bench --bench simd_vs_scalar_benchmark
```

### Run Specific Benchmark Group
```bash
# Only exponential function
cargo bench --bench simd_vs_scalar_benchmark -- Exponential

# Only sine function
cargo bench --bench simd_vs_scalar_benchmark -- Sine

# Only sum reduction
cargo bench --bench simd_vs_scalar_benchmark -- Sum
```

### Run Specific Size
```bash
# Only large arrays (4096 elements)
cargo bench --bench simd_vs_scalar_benchmark -- /4096

# Only medium arrays (512 elements)
cargo bench --bench simd_vs_scalar_benchmark -- /512
```

### Generate HTML Report
```bash
cargo bench --bench simd_vs_scalar_benchmark
# Open target/criterion/report/index.html in browser
```

## Interpreting Results

### Expected Speedup Ranges

**Small Arrays (64 elements)**:
- Speedup: 1.0x - 1.5x
- Reason: Overhead of SIMD setup may dominate for small inputs

**Medium Arrays (512 elements)**:
- Speedup: 2.0x - 4.0x
- Reason: SIMD benefits outweigh setup costs

**Large Arrays (4096 elements)**:
- Speedup: 3.0x - 6.0x
- Reason: Maximum SIMD utilization, good cache behavior

**Huge Arrays (32768 elements)**:
- Speedup: 2.5x - 5.0x
- Reason: Memory bandwidth may limit speedup

### Sample Output

```
Exponential/SIMD/512     time:   [1.234 µs 1.245 µs 1.256 µs]
                         thrpt:  [407.64 Melem/s 411.23 Melem/s 414.82 Melem/s]

Exponential/Scalar/512   time:   [4.567 µs 4.589 µs 4.611 µs]
                         thrpt:  [111.01 Melem/s 111.56 Melem/s 112.11 Melem/s]
```

**Interpretation**:
- SIMD: 1.245 µs for 512 elements = 411 million elements/second
- Scalar: 4.589 µs for 512 elements = 112 million elements/second
- **Speedup: ~3.7x** (4.589 / 1.245)

## Throughput Metrics

All benchmarks report throughput in **Melem/s** (million elements per second):
- Higher is better
- Compare SIMD vs Scalar for same size
- Larger arrays should have higher throughput (up to cache limits)

## Hardware Considerations

### AVX2 Support

SIMD benchmarks require AVX2 support. Check your CPU:
```bash
# Linux/macOS
grep avx2 /proc/cpuinfo  # Linux
sysctl -a | grep avx2    # macOS

# Or use rustc
rustc --print target-features
```

Without AVX2, SIMD code falls back to scalar, showing minimal speedup.

### CPU Frequency Scaling

For consistent results, disable frequency scaling:
```bash
# Linux
sudo cpupower frequency-set --governor performance

# Restore after benchmarking
sudo cpupower frequency-set --governor powersave
```

## Benchmark Implementation Details

### SIMD Path
Uses `numrs2::ufuncs` module which automatically dispatches to SIMD implementations (AVX2) when available:
```rust
use numrs2::ufuncs;
let result = ufuncs::exp(&array);  // Uses AVX2 if available
```

### Scalar Path
Pure Rust implementation without vectorization:
```rust
fn scalar_exp(arr: &Array<f64>) -> Array<f64> {
    let data = arr.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| x.exp()).collect();
    Array::from_vec(result)
}
```

## Adding New Benchmarks

To add a new operation to the benchmark suite:

1. **Add Helper Function** (if needed):
```rust
fn scalar_new_op(arr: &Array<f64>) -> Array<f64> {
    let data = arr.to_vec();
    let result: Vec<f64> = data.iter().map(|&x| /* operation */).collect();
    Array::from_vec(result)
}
```

2. **Add Benchmark Group**:
```rust
fn bench_new_op(c: &mut Criterion) {
    let mut group = c.benchmark_group("NewOperation");

    for size in [SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE, HUGE_SIZE] {
        group.throughput(Throughput::Elements(size as u64));
        let data = random_f64_array(size);

        group.bench_with_input(BenchmarkId::new("SIMD", size), &data, |b, data| {
            b.iter(|| black_box(ufuncs::new_op(data)));
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), &data, |b, data| {
            b.iter(|| black_box(scalar_new_op(data)));
        });
    }

    group.finish();
}
```

3. **Register in criterion_group!**:
```rust
criterion_group!(
    benches,
    bench_exp,
    // ... existing benchmarks ...
    bench_new_op  // Add here
);
```

## Performance Tips

### Maximize SIMD Benefits

1. **Array Alignment**: SIMD works best with 32-byte aligned arrays
2. **Contiguous Memory**: Ensure arrays are contiguous in memory
3. **Batch Operations**: Process multiple operations together
4. **Avoid Branching**: Minimize conditional logic in hot loops

### When SIMD May Not Help

- Very small arrays (< 32 elements)
- Operations with heavy branching
- Memory-bound operations (dominated by memory access)
- Functions with complex dependencies between elements

## Related Benchmarks

- `core_operations_benchmark.rs`: General array operations
- `linear_algebra_benchmark.rs`: Matrix operations and decompositions
- `special_functions_benchmark.rs`: Special mathematical functions
- `production_readiness_benchmark.rs`: Real-world usage patterns

## References

- [Intel AVX2 Programming Reference](https://software.intel.com/sites/landingpage/IntrinsicsGuide/)
- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [NumRS2 SIMD Architecture](../docs/ARCHITECTURE.md#simd-optimization)

## Contributing

When adding SIMD optimizations:
1. Add benchmarks to this suite
2. Verify speedup is significant (>1.5x for large arrays)
3. Test on multiple array sizes
4. Document any special requirements or limitations
