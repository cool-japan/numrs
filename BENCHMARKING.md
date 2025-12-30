# NumRS2 Benchmarks

Benchmarking NumRS2 with Criterion.

## Usage

NumRS2 uses [Criterion](https://github.com/bheisler/criterion.rs) for benchmarking, a statistics-driven benchmarking library for Rust. Before running benchmarks, ensure that you have the development version of NumRS2 built.

To run all benchmarks, navigate to the root NumRS2 directory at the command line and execute:

```bash
cargo bench
```

This builds NumRS2 and runs all available benchmarks defined in the `benches/` directory. Note that this could take a while as each benchmark is run multiple times to measure the distribution in execution times.

For **testing** benchmarks locally during development, you can run specific benchmarks:

```bash
cargo bench --bench array_ops
```

Where `array_ops` is the name of a specific benchmark file in the `benches/` directory.

To run a specific benchmark within a benchmark file:

```bash
cargo bench --bench array_ops -- add
```

This runs only benchmarks with names containing "add" in the `array_ops` benchmark file.

## Viewing Results

Criterion automatically generates HTML reports for benchmarks. After running benchmarks, you can view these reports in the `target/criterion/report/index.html` file:

```bash
# On Linux
xdg-open target/criterion/report/index.html

# On macOS
open target/criterion/report/index.html
```

## Comparing Changes

Criterion automatically compares benchmark results to the previous run. To compare against a specific baseline:

1. Run the benchmarks on your baseline branch/commit:
   ```bash
   cargo bench
   ```

2. Save the baseline results:
   ```bash
   cp -r target/criterion baseline_criterion
   ```

3. Make your changes, then run the benchmarks again:
   ```bash
   cargo bench
   ```

4. Compare with your saved baseline:
   ```bash
   critcmp baseline_criterion target/criterion
   ```

   Note: You may need to install `critcmp` first with `cargo install critcmp`.

## Writing Benchmarks

See [Criterion documentation](https://bheisler.github.io/criterion.rs/book/index.html) for basics on how to write benchmarks.

### Example Benchmark

Here's an example benchmark for array addition:

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use numrs2::prelude::*;

fn bench_array_add(c: &mut Criterion) {
    // Setup code
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);
    
    // Benchmark the operation
    c.bench_function("array_add_2x2", |bencher| {
        bencher.iter(|| a.add(&b))
    });
}

criterion_group!(benches, bench_array_add);
criterion_main!(benches);
```

### Best Practices

When writing benchmarks:

1. **Separate setup from measurement**: Prepare arrays and other data in the setup code, not in the benchmarked code block.

2. **Mind memory allocation**: Be aware that operations creating new arrays involve memory allocation, which can affect benchmark results.

3. **Consider realistic scenarios**: Benchmark with array sizes and operations that reflect real-world usage patterns.

4. **Control for page faults**: Initialize arrays with actual values to ensure memory is physically allocated before benchmarking.

5. **Parameterize benchmarks**: Use Criterion's capability to benchmark across different parameter values (e.g., array sizes).

6. **Benchmark against comparable libraries**: Where applicable, include benchmarks comparing NumRS2 to other libraries like ndarray.

7. **Document what's being measured**: Each benchmark should have a clear description of what operation is being measured.

## Performance Tracking

Performance regressions should be taken seriously. Before merging pull requests that might affect performance:

1. Run benchmarks before and after changes
2. Document any performance changes in the PR description
3. If a performance regression is expected, explain why the tradeoff is necessary

## Benchmark Organization

Benchmarks are organized by functional area, similar to the codebase organization:

- `array_ops.rs`: Basic array operations
- `linalg.rs`: Linear algebra operations
- `random.rs`: Random number generation
- `simd.rs`: SIMD-optimized operations
- And so on...

## Benchmark Infrastructure

Future development will include CI-based benchmark tracking to automatically identify performance regressions across commits.