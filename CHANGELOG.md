# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-08-18

### Added
- `Array::as_slice()` / `Array::as_slice_mut()` — zero-copy contiguous-slice access; avoids heap
  allocation on the common C-contiguous layout (falls back to `to_vec()` for non-contiguous arrays)
- `Array::as_cow_1d()` (crate-internal) — zero-copy `CowArray<T, Ix1>` view for SIMD hot-paths in
  `simd.rs`, `ufuncs.rs`, and `linalg.rs`
- `patches/alloca/` — pure-Rust drop-in for the `alloca 0.4.0` crate (C VLA-based original causes
  linker errors on aarch64-apple-darwin with criterion 0.8.2); wired via `[patch.crates-io]`
- `bench/allocation_hotpath_benchmark.rs` — criterion benchmarks for `map`, `zip_with`, `sum`,
  `ufuncs::hypot`, and `simd_add` hot-paths (measures the zero-copy refactor impact)
- `[[example]] visualization` — visualization example is now fully activated with a proper
  `required-features = ["visualization"]` entry in Cargo.toml
- `QuantumCircuit::controlled_u` / `controlled_u_gate<T>` — generic multi-control gate accepting
  any 2^k × 2^k unitary; emits block-diagonal diag(I, U) of size 2^(k+m) with unitarity check
- `QuantumCircuit::multiply_square_gates`, `embed_gate_in_space`, `fuse_adjacent_gates` — general
  n×n gate fusion with kron-embedding onto the union of qubit supports
- `ArrayView::iter()` restored with correct `ArrayViewIterator<'_, T>` lifetime annotation;
  five new unit tests cover round-trip iteration

### Changed
- `scirs2-core`, `scirs2-stats`, `scirs2-linalg`, `scirs2-ndimage`, `scirs2-spatial`,
  `scirs2-special`, `scirs2-fft` updated from 0.5.0 → 0.6.5; these workspace dependencies now use
  `{ workspace = true }` (workspace policy compliance; versions still pinned at workspace root)
- `oxiarc-archive` and `oxiarc-lz4` updated from 0.3.2 → 0.4.1
- `oxicode` updated from 0.2.4 → 0.2.6
- `wgpu` updated 29.0.3 → 30.0.0 (upstream `BufferSlice::get_mapped_range()` now returns
  `Result` instead of panicking internally; `RequestAdapterOptions` gained `apply_limit_buckets`);
  `pyo3` 0.28 → 0.29; `wasm-bindgen` 0.2.125 → 0.2.126, `wasm-bindgen-futures` 0.4.75 → 0.4.76,
  `js-sys` / `web-sys` 0.3.102 → 0.3.103, `wasm-bindgen-test` 0.3.75 → 0.3.76;
  `memmap2` 0.9.10 → 0.9.11
- Random-number generation migrated from `scirs2_stats::distributions` to `scirs2_core::random`
  (`Rng::sample` / `Rng::random_range`); `NonCentralChiSquared::sample`, `NonCentralF::sample`,
  `VonMises::sample`, `Maxwell::sample`, `Wald::sample` (`numrs2::random::advanced_distributions`)
  now take an explicit `&mut StdRng` parameter instead of drawing from an internal `thread_rng()`
  call, improving reproducibility from a seeded generator — **breaking** for direct callers of
  these `.sample()` methods; the crate-level `noncentral_chisquare()` / `vonmises()` / `maxwell()`
  / `wald()` functions keep their existing public signatures
- `bench/bench_distributions.rs`: `f_dist` and `multivariate_normal_cholesky` benchmarks
  re-enabled; removed stale TODO stubs
- `bench/stats_benchmarks.rs`: `bench_statistical_moments` (skewness, kurtosis) and
  `bench_random_sampling` benchmarks re-enabled; fixed `shuffle`/`choice` import path
- `examples/visualization.rs`: stub `main` replaced with full implementation; updated
  `.sample(&mut rng)` → `rng.sample(...)` for modern `rand_distr` API

### Fixed
- **erfinv / erfcinv precision** (`src/new_modules/special/error_functions.rs`): replaced
  previous approximation with Winitzki 2008 initial guess + 8 Halley iterations; absolute error
  reduced to < 1e-9 (previously ~1e-5)
- **Bessel Y_n** (`src/new_modules/special/bessel.rs`): rewrote `bessel_y_scalar` with DLMF
  10.8.1 series for small x and upward recurrence `Y_{n+1} = (2n/x)Y_n − Y_{n-1}`; previous
  code divided by `sin(nπ) = 0` for integer orders, returning NaN
- **Bessel K_n** (`src/new_modules/special/bessel.rs`): rewrote `bessel_k_scalar` with DLMF
  10.31.1/10.31.2 series + upward recurrence; previous code had a dead `8·0 = 0` branch for K₀
  and an inaccurate K₁ formula; both now achieve < 1e-8 absolute error
- **Incomplete gamma tightened** (`src/new_modules/special/gamma.rs`): re-verified
  series + Lentz CF implementation; tightened test assertions from ±0.1 to ~1e-9
- **Elliptic integrals tightened** (`src/new_modules/special/elliptic.rs`): lowered AGM ε
  toward machine epsilon; tightened test assertions from 10% to ~1e-10 abs-diff
- **irfft / irfft2 Hermitian reconstruction** (`src/new_modules/fft.rs`): fixed `mirror_start`
  (even n: start at bin 2, odd n: start at bin 1) so `irfft(rfft(x)) ≈ x` holds for all signal
  lengths; replaced silent `NumCast::unwrap_or(zero)` twiddle conversions with proper casts
- **`split` axis=1** (`src/lib.rs`, `src/array_ops/splitting.rs`): re-enabled previously-disabled
  integration test; verified `ndarray::Axis` slicing works for all axes
- **Numerical-stability test assertions** (`tests/test_special_reference.rs`): replaced four
  stale TODO-gated range checks with tight abs-diff assertions against scipy-verified reference
  values (erf, erfc, gamma, bessel_k)
- **Schur reconstruction** (`tests/test_linalg_reference.rs`): re-enabled A = Q·T·Qᵀ assertion
  at TOLERANCE = 1e-8 (was silently assigned to `_`)
- **Criterion / alloca linker error on Apple Silicon**: `patches/alloca/` pure-Rust replacement
  eliminates C VLA anonymous-symbol incompatibility with aarch64 rustc; all criterion benchmarks
  now build and run on Apple M-series hardware
- **Gamma distribution scale parameter** (`src/random/legacy.rs`, `src/random/state.rs`,
  `src/random/generator.rs`): removed the `1/scale` double-inversion workaround for a since-fixed
  upstream `scirs2_core::Gamma` scale bug; gamma sampling now passes `scale` directly to
  `Gamma::new`
- **RNG seeding reproducibility**: `Generator::random()` now seeds a per-call RNG from the bit
  generator stream instead of drawing directly from a `Uniform` distribution, so output is
  reproducible from the global seed; re-enabled 8 previously-`#[ignore]`d seeding tests across
  `src/random/distributions.rs` (`test_set_seed`) and `tests/test_complex_seed_scenario.rs`,
  `tests/test_global_vs_direct_vonmises.rs`, `tests/test_random_advanced_distributions.rs` (×2),
  `tests/test_random_properties.rs`, `tests/test_random_state.rs`,
  `tests/test_vonmises_randomness_consumption.rs`, `tests/test_vonmises_seed_issue.rs`
- **GPU buffer-mapping error handling** (`src/gpu/array.rs`, `src/gpu/ops.rs`,
  `src/gpu/context.rs`): replaced `.expect()` panics on `wgpu` device-poll/buffer-mapping failures
  with proper `Result` / `NumRs2Error::RuntimeError` propagation, required by the `wgpu` 30.0.0
  API surface (`get_mapped_range()` is now fallible) and consistent with the no-`unwrap()`-in-
  production policy
- **Flaky global-RNG test races**: 11 test files (`src/random/distributions.rs`,
  `advanced_distributions.rs`, `distributions_enhanced.rs`; `tests/test_random_advanced.rs`,
  `test_distribution_reference.rs`, `test_random_advanced_distributions.rs`,
  `test_random_statistical.rs`, `test_random_reference.rs`, `test_random_properties.rs`,
  `test_scirs_integration.rs`, `test_scirs_reference.rs`) had multiple `#[test]` functions calling
  `set_seed()` and sampling the shared `GLOBAL_RANDOM_STATE` with no `#[serial]` isolation, so
  parallel test execution could let one test's reseed corrupt another's in-flight deterministic
  sequence (observed: `test_uniform_sum_to_approximate_normal` intermittently failing a
  goodness-of-fit assertion under `cargo nextest run`, passing reliably in isolation); added
  `#[serial]` (existing `serial_test` dev-dependency, precedented in `src/error_handling.rs`) to
  every affected test function

## [0.4.0] - 2026-06-05

### Added
- **Autodiff** (`src/autodiff/`): forward/reverse mode automatic differentiation with higher-order support (Hessian, Jacobian, hyperdual, Taylor)
- **Distributed computing** (`src/distributed/`): data/model parallelism framework, collectives, distributed optimizers, coordinator (~8,000 lines)
- **Visualization** (`src/viz/`): plotters-based 2D/3D/matrix/stats/performance plots (pure Rust)
- **WebAssembly bindings** (`src/wasm/`): array, linalg, stats, utils bindings; JS/Vite demo app in `examples/wasm/`
- **Reinforcement learning** (`src/new_modules/rl/`): DQN, Actor-Critic, PPO agents, prioritized replay, environment wrappers
- **Quantum computing** (`src/new_modules/quantum/`): gates, circuits, state vector simulation, measurements, Deutsch-Jozsa, Grover, QFT
- **Model I/O & serving** (`src/new_modules/model_io/`, `src/new_modules/serving/`): ONNX-compatible serialization, inference engine, model registry, real-time monitoring
- **CMA-ES optimizer** (`src/optimize/cma_es/`): IPOP-CMA-ES with step-size adaptation, rank-μ/rank-one covariance updates
- **Bayesian optimization** (`src/optimize/bayesian_opt.rs`): GP surrogate, EI/PI/UCB acquisition functions, Matérn/RBF kernels
- **Computer vision** (`src/new_modules/cv/`): Gaussian/bilateral/median filters, Canny edges, Harris/FAST corners, morphological ops
- **Computational geometry** (`src/new_modules/geometry/`): convex hull (Graham scan), Delaunay triangulation, Voronoi diagrams, polygon ops
- **FEM solver** (`src/new_modules/fem/`): 1D/2D FEM with Dirichlet/Neumann BCs, Gaussian quadrature, direct/iterative solvers
- **Wavelets** (`src/new_modules/wavelets/`): DWT/CWT, Haar, Daubechies (db2–db10), Symlet, Coiflet families
- **Graph algorithms** (`src/new_modules/graph/`): BFS/DFS, Dijkstra/A*/Floyd-Warshall, MST (Kruskal/Prim), max-flow (Dinic)
- **Information theory** (`src/new_modules/information_theory/`): Shannon entropy, mutual information, KL/JS divergence, channel capacity
- **Control systems** (`src/new_modules/control/`): transfer functions, state space, Routh-Hurwitz, Bode/Nyquist, PID/LQR/pole placement
- **Physical constants** (`src/new_modules/constants/`): CODATA 2018/2022 constants with uncertainties, unit conversions
- **`average_with_weights`** in `src/stats/basic.rs`: public function returning `(weighted_avg, sum_of_weights)` tuple (NumPy `returned=True` semantics)
- **`skew()` and `kurtosis()`** in `src/math/statistics.rs` and `src/math/mod.rs`: compute sample skewness (Fisher-Pearson) and excess kurtosis (fourth standardised moment minus 3); both accept an optional `axis` parameter and are re-exported from `prelude`
- **`f_dist()`** in `src/random/distributions.rs`: sample from the F distribution (ratio of two chi-squared variates) given numerator and denominator degrees of freedom
- **`instance_norm()`** in `src/nn/normalization.rs`: instance normalisation for 2-D tensors (normalises each row independently with configurable epsilon)
- **Python optimizer methods** (`src/python/optimize.rs`): `py_minimize` now supports `"bfgs"` method (numerical gradient via central differences) in addition to the existing `"nelder-mead"` default
- **VECM fitting via Johansen procedure** (`src/new_modules/timeseries/var.rs`): `VecmModel::fit` now implements the full Johansen (1988) cointegration/error-correction estimation instead of a placeholder
- **FEM 2D point evaluation** (`src/new_modules/fem/solvers.rs`): FEM solver can now evaluate solutions at arbitrary 2D points using element shape functions

### Changed
- `scirs2-core`, `scirs2-stats`, `scirs2-linalg`, `scirs2-ndimage`, `scirs2-spatial`, `scirs2-special`, `scirs2-fft`, `scirs2-numpy` updated from 0.4.2 → 0.5.0
- `oxiarc-archive` and `oxiarc-lz4` updated from 0.2.6 → 0.3.2
- `oxicode` updated from 0.2 → 0.2.4

### Fixed
- **Build fix**: Updated `oxiarc-core` lockfile resolution from 0.2.6 to 0.2.8; resolves `oxiarc_core::cancel`/`progress` missing-module compile errors introduced by `scirs2-core 0.4.4`'s transitive `oxiarc-lz4 0.2.8`/`oxiarc-zstd 0.2.8` dependencies
- **Stable SVD for large matrices** (`src/linalg_stable.rs`): `svd_bidiagonal` now runs full Golub–Kahan bidiagonalization + Jacobi SVD instead of silently falling back to the n≤3 path
- **Stable eigendecomposition for large matrices** (`src/linalg_stable.rs`): `symmetric_eigen_tridiagonal` now runs a cyclic Jacobi sweeps algorithm instead of silently falling back to the n≤3 path
- **Quantum partial trace** (`src/new_modules/quantum/statevector.rs`): correct bit-interleaving index reconstruction; previous code used `full_i = i` (identity mapping), producing a wrong density matrix for all multi-qubit traces
- **Polynomial domain mapping** (`src/new_modules/polynomial/utils.rs`): `polyscale` now performs real polynomial composition under the affine map; previously returned the input coefficients unchanged
- **FEM matrix operations for n>3** (`src/new_modules/fem/elements.rs`): `matrix_determinant` and `matrix_inverse` now handle arbitrary n×n via LU with partial pivoting; previously hard-errored for n>3, blocking 3D/higher-order FEM elements
- **Schur decomposition** (`src/new_modules/matrix_decomp/schur.rs`): replaced single Rayleigh-shift with Francis implicit double-shift QR + deflation for correct real Schur form including complex-conjugate eigenpair blocks
- **Boltzmann exploration** (`src/new_modules/rl/utils.rs`): `BoltzmannExploration::select_action` now computes temperature-scaled softmax over Q-values; previously fell back to greedy action selection regardless of temperature
- **`gamma_ln` accuracy** (`src/new_modules/probabilistic/distributions.rs`): replaced custom Stirling approximation with `scirs2_special::loggamma` (Lanczos g=7, ~15-digit accuracy) with reflection formula for x<0.5
- **Unsafe block warnings** (`src/memory_optimize/cache_layout.rs`): removed unnecessary `unsafe {}` wrappers around `std::arch::x86_64::__cpuid` (now a safe function in current Rust)
- **Clippy compliance** (`src/simd_optimize/avx2_enhanced/`): replaced manual copy loops with `copy_from_slice`; replaced `vec![...]` with array literals in tests

### Notes
- **WASM blocker resolved**: `scirs2-spatial 0.4.4` feature-gates tokio under `async` (not enabled by default); `numrs2` uses only the `parallel` feature, so no transitive tokio. Browser WASM (`wasm32-unknown-unknown`) still requires disabling the `gpu` and `distributed` features (which pull tokio directly)
- **Known future work**: `distributed/collective.rs` operations (scatter/gather/all-reduce) are stub-implemented returning empty results pending a real network transport layer; `distributed/linalg.rs` distributed linear algebra returns `NotImplemented`; GPU `NotImplemented` branches in `gpu/batching.rs` (Conv2D batching) and `gpu/ops.rs` (N-D broadcasting)

## [0.3.3] - 2026-04-18

### Added
- **New Benchmarks**: Added comprehensive benchmarks for I/O operations (`bench/io_benchmarks.rs`), complex number operations (`benches/complex_benchmark.rs`), and sparse matrix operations (`benches/sparse_benchmark.rs`)
- **Distributed Optimization**: Enhanced distributed optimization module in `src/distributed/optimization.rs`

### Changed
- **Dependency Upgrades**: Updated all dependencies to latest versions in Cargo.toml

### Fixed
- **Linter Compliance**: Resolved clippy warnings in benchmark files and distributed optimization module

## [0.3.2] - 2026-03-27

### Changed
- **Version Bump**: Updated to v0.3.2 patch release
- **PyPI Compatibility**: Improved PyPI publishing configuration

### Fixed
- **MOS (Minimum Output Size)**: Resolved minimum output size constraints

## [0.3.1] - 2026-03-21

### Fixed
- **Clippy Warnings**: Resolved all 24 clippy warnings for MSRV compatibility
  - Fixed `Color` trait ambiguity in viz modules (`matrix.rs`, `perf.rs`, `plot2d.rs`, `plot3d.rs`, `stats.rs`) by adding explicit `use plotters::style::Color` imports
  - Replaced explicit counter loops with idiomatic `enumerate`/`zip` pattern in `src/cluster.rs`
  - Replaced manual checked division patterns with `.checked_div()` in `performance_tuning.rs`, `access_patterns.rs`, and `scheduler.rs`

## [0.3.0] - 2026-03-06

### Changed
- **SciRS2 Ecosystem Update**: Updated all scirs2-* dependencies to v0.3.0
  - scirs2-core v0.3.0: Latest core with enhanced SIMD, parallel, random operations
  - scirs2-linalg v0.3.0: Linear algebra improvements with OxiBLAS
  - scirs2-stats v0.3.0: Statistical functions enhancements
  - scirs2-fft v0.3.0: FFT operations improvements
  - scirs2-ndimage v0.3.0: N-dimensional image processing updates
  - scirs2-spatial v0.3.0: Spatial algorithms with improved KD-trees
  - scirs2-special v0.3.0: Special functions updates
  - scirs2-numpy v0.3.0: Python bindings compatibility updates
- **NPZ Compression**: Enabled DEFLATE compression for .npz files (OxiARC v0.3.0+ multi-file bug fixed)
- **Cyclic Spline Solver**: Replaced O(n²) Gaussian elimination with Sherman-Morrison O(n) cyclic Thomas algorithm

### Fixed
- Fixed WASM version assertion tests to use "0.3" instead of "0.2"

## [0.2.0] - 2026-01-30

### Changed
- **COOLJAPAN Ecosystem Compliance**: Full compliance with COOLJAPAN pure Rust policies
  - Replaced `numpy` dependency with `scirs2-numpy` (v0.3.0) for Python bindings
  - Removed OpenBLAS linker flags from `.cargo/config.toml` (now using OxiBLAS pure Rust backend)
  - Removed `cdylib` crate-type (Python extension builds handled by maturin)

### Fixed
- Fixed linking errors when building with `--all-features` due to openblas flags
- Fixed Python symbol resolution issues in test builds

### Dependencies
- **scirs2-numpy**: v0.3.0 (replaces direct numpy dependency)
- **SciRS2 Ecosystem**: scirs2-* v0.3.0 (latest stable releases)
- All Python bindings now go through SciRS2 ecosystem

## [0.1.1] - 2025-12-30

### Added
- **Initial Release**: First stable release of NumRS2, a NumPy-inspired numerical computing library for Rust
- **Core Array Operations**: Comprehensive ndarray-like API with multi-dimensional arrays
  - Array creation, manipulation, and reshaping operations
  - Broadcasting support for element-wise operations
  - Advanced indexing (fancy indexing, boolean masking, multi-dimensional slicing)
  - Zero-copy views and efficient memory management

- **Expression Templates**: Lifetime-free expression template system for lazy evaluation
  - SharedArray<T> with reference-counted storage for O(1) cloning
  - Operator overloading for natural syntax (+, -, *, /, scalar operations)
  - Common Subexpression Elimination (CSE) for automatic optimization
  - Cache-aware memory access patterns for improved performance

- **SIMD Optimization**: Comprehensive vectorization support
  - AVX2/AVX512 support for x86_64 architectures
  - ARM NEON support for ARM architectures
  - Automatic threshold-based dispatch between SIMD and scalar implementations
  - 86+ optimized functions with 4-way loop unrolling and FMA instructions

- **Linear Algebra**: Complete linear algebra stack
  - Matrix operations (multiplication, transpose, inverse, determinant)
  - Decompositions (SVD, QR, LU, Cholesky, Eigenvalue)
  - Iterative solvers (Conjugate Gradient, GMRES, BiCGSTAB)
  - Randomized algorithms for large-scale computations
  - Sparse matrix support (COO, CSR, CSC, DIA formats)

- **Mathematical Functions**: Extensive mathematical operations
  - Trigonometric, hyperbolic, exponential, logarithmic functions
  - Special functions (gamma, beta, error functions, Bessel functions)
  - Polynomial operations (evaluation, fitting, root finding)
  - Cubic spline interpolation with multiple boundary conditions

- **Statistical Functions**: Comprehensive statistical toolkit
  - Descriptive statistics (mean, median, variance, standard deviation)
  - Distribution functions and random number generation
  - Hypothesis testing and correlation analysis
  - Integration with SciRS2 statistical modules

- **Signal Processing**: FFT and filtering operations
  - Fast Fourier Transform (FFT/IFFT)
  - Convolution and correlation
  - Digital filtering operations

- **Interoperability**: Multiple data format support
  - NumPy format (.npy, .npz) for Python compatibility
  - Apache Arrow integration for zero-copy data exchange
  - CSV and binary serialization support
  - Memory-mapped file I/O

- **Financial Computing**: Financial analysis tools
  - Options pricing models
  - Bond valuation
  - Time value of money calculations
  - Financial metrics and indicators

- **Automatic Differentiation**: Forward and reverse mode AD
  - Dual numbers for forward mode
  - Tape-based backpropagation for reverse mode
  - Higher-order derivatives (Hessian, Taylor series)

- **SciRS2 Ecosystem Integration**: Built on the SciRS2 scientific computing foundation
  - scirs2-core v0.3.0: SIMD, parallel, random, array operations
  - scirs2-linalg v0.3.0: Linear algebra with OxiBLAS
  - scirs2-stats v0.3.0: Statistical functions
  - scirs2-fft v0.3.0: FFT operations
  - scirs2-signal v0.3.0: Signal processing
  - scirs2-special v0.3.0: Special functions
  - scirs2-ndimage v0.3.0: N-dimensional image processing
  - scirs2-spatial v0.3.0: Spatial algorithms

### Technical Details
- **Total Rust Code**: ~155,000 lines of production-ready code
- **Test Coverage**: 1,111+ unit tests passing, comprehensive test suite
- **Quality Metrics**: Zero compilation warnings, zero clippy errors
- **Performance**: SIMD-optimized operations with automatic fallback
- **Pure Rust**: No C/C++ dependencies, built on OxiBLAS (pure Rust BLAS/LAPACK)

### Dependencies
- **SciRS2 Ecosystem**: scirs2-* v0.3.0 (stable releases)
- **OxiBLAS**: v0.3.0 (pure Rust BLAS/LAPACK implementation)
- **Oxicode**: v0.3.0 (pure Rust serialization)
- All dependencies use stable, production-ready versions

This initial release provides a comprehensive NumPy-like experience in Rust with production-ready quality, extensive test coverage, and pure Rust dependencies for maximum portability and safety.

[0.4.1]: https://github.com/cool-japan/numrs/releases/tag/v0.4.1
