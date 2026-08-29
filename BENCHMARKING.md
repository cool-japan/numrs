# NumRS2 Benchmarks

NumRS2 uses [Criterion](https://github.com/bheisler/criterion.rs) for benchmarking. This document
lists the current bench targets (from `Cargo.toml`'s `[[bench]]` table) and, most importantly, the
methodology this project actually uses to get a trustworthy number on a shared, often-loaded
development machine.

## Running benchmarks

```bash
cargo bench                              # every bench target
cargo bench --bench matmul_dispatch_benchmark   # one target
cargo bench --bench matmul_dispatch_benchmark -- gemm_2d   # filter by name within a target
cargo bench --features distributed --bench distributed_benchmarks  # feature-gated targets
```

Criterion writes HTML reports to `target/criterion/report/index.html`
(`open target/criterion/report/index.html` on macOS, `xdg-open` on Linux).

## The min-over-alternating-A/B methodology

**The problem this solves:** development machines here are frequently shared with other
concurrent work (other agents, other builds). Criterion reports a *mean* per benchmark, and a
neighbouring process descheduling one candidate for a few hundred milliseconds inflates that
candidate's mean while leaving the other alone — which is exactly how a real 3% regression gets
reported as a 40% one, or how a real win gets hidden entirely. This is not a hypothetical: it is
the documented reason two of this project's own bench targets ship a dedicated alternate mode.

**The fix:** instead of running candidate A for N seconds and then candidate B for N seconds (so
each is exposed to a *different* slice of whatever else is happening on the machine), interleave
them — A, B, A, B, ... — within each measurement round, and keep the **minimum** sample observed
for each candidate across all rounds, not the mean. The minimum is far harder to bias upward by
background load (a competing process can only ever make a sample slower, never faster), and
interleaving means both candidates see the *same* slices of contention, so a true difference
survives even under load while a load-induced difference washes out.

**Where it's built in:**

```bash
COW_AB_REPORT=1 cargo bench --bench cow_mutation_guard      # src/array/core.rs COW guard cost
EXPR_AB_REPORT=1 cargo bench --bench expr_fused_benchmark   # fused .eval() vs. eager
```

Both env-var modes are also the only way to see the *pre-refactor* baseline for their respective
change (COW-guarded `Array` / fused expressions), since the old code path no longer exists in the
tree and has to be reconstructed for comparison (see `precow_set` in `bench/cow_mutation_guard.rs`).

**Where the same discipline is applied manually:** the performance tables embedded in
`src/kernels/gemm.rs` and `src/kernels/reduce.rs`'s own module docs were produced the same way —
alternating/round-robin runs of the candidates being compared, keeping the load-matched
observation rather than a raw mean — even though those two files don't (yet) have a dedicated
`_AB_REPORT` env var of their own. Read the surrounding doc comment before trusting a number from
either file: several call out which rows were measured under background load and which were not,
and say so explicitly rather than presenting every row as equally clean.

**Practical rule of thumb:** on this machine, treat any single `cargo bench` run's mean as a
rough signal, not a verified number. For anything going into a changelog or a PR description,
either use an `_AB_REPORT` mode where one exists, or run the comparison at least twice and report
the number that recurs, not the first one you see.

## Bench targets

Most targets live in `bench/` (with an explicit `path =` override in `Cargo.toml`); a handful of
older ones remain in Cargo's default `benches/` directory. All use the `criterion` harness
(`harness = false`) except `numpy_comparison`, which uses the default libtest harness.

### Dispatch-layer benchmarks (`src/kernels/`)

| Target | What it measures |
|---|---|
| `matmul_dispatch_benchmark` | `kernels::gemm`'s 2D matmul dispatch table across shapes/dtypes vs. the blocked-serial, SIMD-serial, and `blas_accelerated` alternatives |
| `elementwise_dispatch_benchmark` | `kernels::elementwise`'s unary/binary dispatch thresholds |
| `reduction_dispatch_benchmark` | `kernels::reduce`'s sum/mean/var/min/max dispatch thresholds and parallel-tier crossover points |
| `cow_mutation_guard` | The `Arc::make_mut` uniqueness-check cost on `Array<T>`'s copy-on-write path — absolute per-call cost when unique, one-time copy cost after a clone, and the bulk-acquisition mitigation. Supports `COW_AB_REPORT=1`. |
| `expr_fused_benchmark` | `IntoExpr::expr()` + `.eval()` fused evaluation vs. the eager equivalent. Supports `EXPR_AB_REPORT=1`. |
| `allocation_hotpath_benchmark` | `map`/`zip_with`/`sum`/`ufuncs::hypot`/`simd_add` zero-copy accessor paths (`as_slice()`/`as_cow_1d()`) vs. a `to_vec()` baseline |

### Core operations

| Target | What it measures |
|---|---|
| `numpy_comparison` (`bench/bench.rs`) | Baseline libtest-harness comparison scaffold |
| `numpy_comparison_benchmark` | Broader NumPy-equivalent operation comparisons |
| `core_operations_benchmark` | Core `Array` operations |
| `array_ops_benchmarks` | `array_ops` module operations |
| `simd_vs_scalar_benchmark`, `simd_comparison_benchmark` | SIMD vs. scalar paths |
| `expression_templates_benchmark` | `SharedArray`/`SharedExpr` lazy-evaluation templates (the pre-existing system, distinct from the `kernels`-era fusion above) |
| `parallel_benchmarks`, `parallel_algorithms_benchmark` (`benches/`) | Parallel execution strategies and algorithms |
| `memory_benchmarks`, `memory_optimization_benchmark` (`benches/`), `cache_alignment_benchmark` (`benches/`) | Memory layout, allocation, and cache-alignment strategies |

### Domain-specific

| Target | What it measures |
|---|---|
| `linalg_benchmarks` | Linear algebra operations and decompositions |
| `stats_benchmarks` | Statistical functions, including `bench_statistical_moments` (skewness/kurtosis) and `bench_random_sampling` |
| `bench_distributions` | Random distribution sampling, including `f_dist` and `multivariate_normal_cholesky` |
| `fft_benchmarks` | FFT operations |
| `optimization_benchmarks`, `multi_objective_benchmark` (`benches/`) | Optimization algorithms, including multi-objective (NSGA-II/III) |
| `nn_benchmarks` | Neural-network layer operations |
| `sparse_benchmark` (`benches/`) | Sparse matrix formats and operations |
| `complex_benchmark` (`benches/`) | Complex-number arithmetic |
| `io_benchmarks` | Serialization/deserialization (NPY/NPZ and friends) |
| `distributed_benchmarks` | Distributed collectives/linear algebra — requires `--features distributed` |

## Comparing changes

Criterion automatically compares each run to the previous run's saved baseline. To compare against
a specific point in history:

```bash
cargo bench                          # on the baseline branch/commit
cp -r target/criterion baseline_criterion
# ... make your changes ...
cargo bench
cargo install critcmp                # once
critcmp baseline_criterion target/criterion
```

For anything the min-over-alternating-A/B methodology applies to (see above), prefer that mode's
own report over a criterion mean-to-mean comparison.

## Writing benchmarks

See the [Criterion book](https://bheisler.github.io/criterion.rs/book/index.html) for the basics.
Beyond that:

1. **Separate setup from measurement** — build inputs outside the `iter()` closure.
2. **Use `black_box`** on inputs/outputs that the optimizer could otherwise prove unused.
3. **If the machine might be loaded, don't trust a single mean** — see the methodology section
   above; add an `_AB_REPORT`-style mode for anything performance-sensitive enough to end up in a
   changelog.
4. **Document what's being measured**, including which comparisons are load-sensitive — the doc
   comments in `src/kernels/gemm.rs`, `src/kernels/reduce.rs`, `bench/cow_mutation_guard.rs`, and
   `bench/expr_fused_benchmark.rs` are the house style to follow: state the exact table, note which
   rows were measured under contention, and say what a regression would look like.
5. **Feature-gate what needs it** — `distributed_benchmarks` requires `--features distributed`;
   follow that pattern for any new bench target that needs a non-default feature.
