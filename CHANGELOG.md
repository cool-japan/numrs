# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-08-26

This release is a production-hardening pass across the whole crate: a shared compute-dispatch
layer, copy-on-write arrays, a real (not fabricated) distributed transport, a large batch of
NumPy-parity additions, and a long list of correctness fixes found while building all of the
above. Every entry below was checked against the working tree while writing this section; where
a specific number could not be re-derived from a file checked into the repository, that is noted
rather than restated as fact.

### Added

**Compute dispatch & fusion**
- `kernels` module (`src/kernels/`): crate-private, dtype-dispatched compute kernels shared by the
  hot paths that used to each hand-roll their own SIMD/parallel thresholds — `elementwise`
  (unary/binary dispatch), `gemm` (2D matmul dispatch table), `reduce` (sum/mean/var/min/max with
  deterministic accumulation order), `cast` (sound `TypeId`-guarded reinterpretation), `borrow`
  (contiguous-vs-owned operand bridging). Exercised by `bench/{matmul,elementwise,reduction}_dispatch_benchmark.rs`.
- Expression fusion: `IntoExpr::expr()` (`src/expr/owned.rs`) builds a lazy `ExprNode` from an
  `Array`/scalar/expression tree; `.eval()` (`src/expr/fused_eval.rs`) walks it in one fused pass
  instead of materializing every intermediate. Honest about scope: eager syntax (`&a + &b`) does
  not fuse on its own — fusion only happens through the explicit `.expr()...eval()` builder. A
  compile-time `fused!` macro is deferred to 0.6.0 (see `src/expr/owned.rs`'s module doc).

**NumPy-parity additions**
- ufunc machinery (`src/ufunc_ops.rs`): `reduce`, `accumulate`, `outer`, `reduceat`, and `.at()`
  (in-place scatter, `np.add.at`-style) for the binary ufunc objects, plus `where=` variants.
- `fftn`/`ifftn`/`rfftn`/`irfftn` (`src/fft/numpy_parity.rs`) with NumPy's exact `s=`/`axes=`/
  `norm=` parameter conventions, implemented as a correctness wrapper over `scirs2_fft` (see
  Fixed and Known Upstream Issues below).
- `pad`: 6 new modes plus `reflect_type="odd"` for `"reflect"`/`"symmetric"`
  (`src/array_ops/manipulation/pad.rs`).
- `quantile`/`percentile`: the 9 NumPy >= 1.22 Hyndman & Fan methods (`inverted_cdf`,
  `averaged_inverted_cdf`, `closest_observation`, `interpolated_inverted_cdf`, `hazen`, `weibull`,
  `linear` [default], `median_unbiased`, `normal_unbiased`), alongside the 4 pre-existing legacy
  methods (`lower`/`higher`/`nearest`/`midpoint`) (`src/stats/quantile.rs`).
- `histogramdd`, plus `density=` normalization for `histogram`/`histogram2d`/`histogramdd`
  (`src/stats/histogram.rs`).
- `multi_dot`, `tensorsolve`, `tensorinv`, and `norm` ord `-2` / `"nuc"` (`src/linalg/parity.rs`).
- Masked-array completion (`src/masked/`): `std`/`var`/`prod`/`median`/`ptp`/`any`/`all`/
  `argmin`/`argmax`/`sort`/`cumsum`/`dot`/`concatenate`, plus `Sub`/`Div`/comparison ops and
  `axis=` support throughout. `argmin`/`argmax` deliberately return `Err` on a fully-masked lane
  instead of `numpy.ma`'s degenerate (and silently ambiguous) index `0` — see NUMPY_MIGRATION.md.
- Polynomial class family (`src/new_modules/polynomial/classes/`): `Chebyshev`, `Legendre`,
  `Hermite`, `HermiteE`, `Laguerre` (alongside the pre-existing `Power`/`Series` classes).
- Random: `SeedSequence` + `Generator::spawn` (independent child streams), `Philox4x64BitGenerator`,
  `SFC64BitGenerator`, and `Generator::permuted` (`src/random/{generator,seed_sequence,sfc64}.rs`).

**Distributed**
- `Endpoint` point-to-point transport (`src/distributed/net/`): real TCP links, a fixed 56-byte
  `FrameHeader` per message, LZ4 payload compression via `oxiarc-lz4`.
- 8 collective operations plus `v`-variants and `reduce_scatter` (`src/distributed/collective.rs`):
  `reduce`, `allreduce` (+ `allreduce_with`), `reduce_scatter`, `broadcast`, `gather`/`gatherv`,
  `allgather`/`allgatherv`, `scatter`/`scatterv`, `allscatter`, `barrier`.
- `TSQR` (Tall-Skinny QR, `src/distributed/linalg/tsqr.rs`): binary-tree, communication-avoiding
  QR for row-block-distributed matrices, keeping every tree factor so `Q` is applied without ever
  being formed; used by `decomp::distributed_qr`/`distributed_svd`/`distributed_solve`. Requires
  every rank's row block to be at least as tall as the matrix is wide (`m_i >= n`); returns
  `UnsupportedShape` rather than approximating otherwise (the wide case needs CAQR, not yet
  implemented — see TODO.md).
- Block-cyclic-column Cholesky (`src/distributed/linalg/cholesky.rs`) plus a distributed
  forward/back-substitution solve.
- `LocalCluster` test harness (`src/distributed/testing.rs`): drives `world_size` copies of an
  async closure over real loopback TCP, for correctness- and deadlock-testing without a
  multi-process launcher.

**GPU / Python / WASM**
- GPU: `norm_l1`, N-D transpose/broadcast/slicing, and `Conv2D` (`src/gpu/{linalg,ops,batching,conv}.rs`).
- Python: wheel builds via `maturin` (`pyproject.toml`); N-D neural-network bindings and
  `random`/`fft` bindings (`src/python/{nn,random,fft}.rs`).
- WASM: `dlmalloc` as the global allocator behind the `wasm` feature, gated to the `wasm32` target
  family (`src/wasm/utils.rs`, `Cargo.toml`).

**Carried over from the previous 0.4.1 pass**
- `Array::as_slice()`/`as_slice_mut()`: zero-copy contiguous-slice access (falls back to
  `to_vec()` off the fast path); `Array::as_cow_1d()` (crate-internal): zero-copy
  `CowArray<T, Ix1>` view for SIMD hot paths.
- `patches/alloca/`: pure-Rust drop-in for the `alloca` crate, unblocking criterion on
  aarch64-apple-darwin.
- `QuantumCircuit::controlled_u`/`controlled_u_gate<T>` (generic multi-control gate for any
  2^k x 2^k unitary) and `multiply_square_gates`/`embed_gate_in_space`/`fuse_adjacent_gates`
  (general n x n gate fusion).
- `ArrayView::iter()` restored with a correctly-annotated `ArrayViewIterator<'_, T>` lifetime.

### Changed

- **`Array<T>` is now `Arc`-backed copy-on-write.** `Clone` is O(1) (an `Arc` bump instead of a
  deep copy); the first mutation after a clone unshares via the crate's single `Arc::make_mut`
  call site (`src/array/core.rs`, CI-guarded by `scripts/ci-local.sh` to stay exactly one
  `make_mut` + one `Arc::try_unwrap`, both in that file); `Array::is_unique()` exposes the sharing
  state. **Disclosure:** `Array<T>` now requires `T: Send + Sync` to remain `Send`/`Sync` itself
  (an `Arc<T>` transitively needs it), which can newly constrain generic call sites.
- **Matmul backend**: the default `f32`/`f64` dispatch tier now goes through
  `scirs2_core::ndarray::linalg::general_mat_mul` (the pure-Rust `matrixmultiply` crate) instead
  of `scirs2-core`'s `simd_matrix_multiply`, parallelizing above a FLOPs threshold via an
  `M`-only row split (`src/kernels/gemm.rs`). Measured up to 18.82x faster than the pre-existing
  loop at `512^3` and 12.70x at `256^3` in that module's own before/after table, with no
  regressed shape.
- **`lapack` is now a default feature** (`default = ["matrix_decomp", "scirs", "lapack"]`,
  `Cargo.toml`): `det`/`inv`/`svd`/`eig`/`qr`/`cholesky` and the rest of core linalg are now
  reachable on a plain `cargo add numrs2` — previously they compiled out unless a consumer opted
  in explicitly.
- Banned-crate policy is enforced by `deny.toml` (`cargo deny check bans`), with documented, dated
  exceptions (2026-08-19) for the pure-Rust transitive codecs (`flate2`/`brotli`/`snap`/`lz4_flex`)
  that ship only inside the optional `parquet`/`matlab`/`visualization` feature graphs.
- `ReduceOp::And`/`Or` (`src/distributed/collective.rs`) now return an explicit error on float
  types instead of a silent, meaningless fallback; use the new `ReduceOp::reduce_bitwise` for
  integer bitwise AND/OR.
- `scirs2-core`/`-stats`/`-linalg`/`-ndimage`/`-spatial`/`-special`/`-fft`: 0.5.0 → 0.6.5, now via
  `{ workspace = true }`.
- `oxiarc-archive`/`oxiarc-lz4` 0.3.2 → 0.4.1; `oxicode` 0.2.4 → 0.2.6; `wgpu` 29.0.3 → 30.0.0;
  `pyo3` 0.28 → 0.29; `wasm-bindgen` 0.2.125 → 0.2.126 (`wasm-bindgen-futures` 0.4.75 → 0.4.76,
  `js-sys`/`web-sys` 0.3.102 → 0.3.103, `wasm-bindgen-test` 0.3.75 → 0.3.76); `memmap2` 0.9.10 →
  0.9.11.
- Random-number generation migrated from `scirs2_stats::distributions` to `scirs2_core::random`.
  **Breaking** for direct callers of `NonCentralChiSquared::sample`/`NonCentralF::sample`/
  `VonMises::sample`/`Maxwell::sample`/`Wald::sample` (`numrs2::random::advanced_distributions`),
  which now take an explicit `&mut StdRng` instead of drawing from an internal `thread_rng()`; the
  crate-level `noncentral_chisquare()`/`vonmises()`/`maxwell()`/`wald()` functions keep their
  existing signatures.
- `bench/bench_distributions.rs` (`f_dist`, `multivariate_normal_cholesky`) and
  `bench/stats_benchmarks.rs` (`bench_statistical_moments`, `bench_random_sampling`) benchmarks
  re-enabled; `examples/visualization.rs`'s stub `main` replaced with a full implementation.

### Fixed

**Correctness — array/linalg core**
- `tile()`: N-D output shape and data ordering (`src/array_ops/tiling.rs`).
- `moveaxis`: permutation now correct for arbitrary axis pairs (`src/array_ops/axis_ops.rs`).
- `as_strided`: general N-D case, not just the low-dimensional special cases (`src/stride_tricks.rs`).
- `Array::to_vec()`: logical (not physical) element order for permuted / Fortran-layout arrays
  (`src/array/core.rs`).
- `gemm`/`gemv`: transpose-flag combinations (`trans_a`/`trans_b`) now all dispatch to the correct
  shape/stride arithmetic (`src/blas.rs`).
- `einsum`: no longer panics on a HashMap miss for an operand referencing an unmapped index
  (`src/linalg/tensor_ops.rs`).
- `solve_lyapunov`: now correct for non-symmetric `A` (`src/new_modules/control/stability.rs`).
- `lstsq`: no longer errors on every non-square input — full SVD (over-determined) vs. thin SVD
  (under-determined) is now selected correctly (`src/new_modules/matrix_decomp/condition.rs`).

**Correctness — statistics & reductions**
- `Statistics::var`/`std` (`src/stats/basic.rs`) and `ufuncs::var`/`std` (`src/ufuncs.rs`): both
  silently flipped from population to sample variance (`n` vs. `n-1`) at length 64 (an upstream
  SIMD-path artifact). Now population semantics everywhere, matching NumPy's default.
  **Disclosure: this changes the numeric value returned by `var`/`std` on arrays of length >= 64**
  relative to the previous (buggy) behavior.
- `min`/`max`: `scirs2_core::simd_ops::simd_min_element`/`simd_max_element` return a wrong
  *finite* value for some NaN placements — not merely an old NaN-convention difference, a live
  upstream bug, pinned as a tripwire in `src/stats/basic.rs`. `kernels::reduce`'s `min`/`max` no
  longer call them; NaN now propagates the NumPy way everywhere, including `optimized_ops` and
  `nn::simd_ops`.
- `min_along_axis`/`max_along_axis` (`src/stats/basic.rs`): panicked on every axis reduction;
  fixed.
- `maximum`/`minimum` (`src/ufuncs.rs`): NaN-propagation asymmetry between the two functions
  removed — both now propagate NaN symmetrically.

**Correctness — special functions, distributions, dates, I/O**
- `student_t_cdf`: corrected via the regularized incomplete beta function
  (`src/random/distributions_enhanced.rs`).
- Sobol sequences: dimensions above 8 now use real Joe-Kuo direction numbers (search criterion 6)
  up to dimension 40, verified bit-for-bit against unscrambled `scipy.stats.qmc.Sobol` for the
  pinned dimensions (`src/random/distributions_enhanced.rs`).
- `busday`/business-day roll conventions corrected (`src/types/datetime/`).
- NetCDF I/O now writes/reads real NetCDF-3 (`src/io/netcdf.rs`).
- `CustomDType` serialization and `StructuredArray::from_arrays` fixed (`src/types/custom.rs`,
  `src/types/structured.rs`).
- VAR log-likelihood (`src/new_modules/timeseries/var.rs`): a failed inversion of the
  residual-covariance matrix Sigma now propagates as an error instead of silently reporting a
  wrong log-likelihood/AIC/BIC.
- Wavelet SURE threshold (`src/new_modules/wavelets/mra.rs`): corrected risk formula for soft
  thresholding.

**FFT**
- Worked around two confirmed `scirs2_fft::fftn` normalization bugs (`"backward"` not scaling by
  `1/N`; `"ortho"`/`"forward"` not restricting the scaling basis to the requested `axes`) in
  `src/fft/numpy_parity.rs`'s own `fftn`; `rfftn` inherits the fix by construction, and
  `ifftn`/`irfftn` were already correct upstream. See Known Upstream Issues below.
- The existing "FHT" test exercised a Hankel transform, not a Hartley transform, despite its
  name; a real Discrete Hartley Transform (`dht`/`idht`, aliased `hartley_fht`) now exists and is
  what the corrected test checks (`src/fft/mod.rs`).

**Soundness**
- Ad hoc, per-call-site raw-pointer type-punning (e.g. `stats/basic.rs`'s
  `x as *const T as *const f64`, `array/operations_optimized.rs`'s bare `mem::transmute_copy`) is
  now centralized behind one audited module, `kernels::cast`, whose every function proves
  soundness via `TypeId::of::<T>() == TypeId::of::<f64|f32>()` before reinterpreting (documented
  in that module's own Soundness section). `io::npy_npz` and `types::structured` keep their
  pre-existing `type_name`/`TypeId`-based dispatch by deliberate design (their generic call sites
  lack the `T: 'static` bound `TypeId` needs) but gained explicit size-check-before-transmute
  guards of their own.
- `src/array/core.rs`, `src/views.rs`, and `src/arrays/*.rs` contain no `unsafe` blocks at all;
  the raw-pointer type-punning that remains in the crate is confined to `kernels::cast`,
  `io::npy_npz`, and `types::structured` (all three audited as above).

**Distributed**
- Collective operations (`broadcast`/`reduce`/`allreduce`/`gather`/`allgather`/`scatter`/
  `allscatter`/`barrier` and the `v`/`reduce_scatter` variants) previously fabricated their
  results — they returned the caller's own local data with no network transport at all. All of
  them are now real, running over the new `Endpoint` TCP transport (`src/distributed/collective.rs`,
  `src/distributed/net/`). **Scope note:** this is about the collectives, not the older
  `DistributedArray`-based `linalg::distributed_qr`/`svd`/`solve`/`matmul`/`matvec`, which
  permanently return `NotImplemented` by design in favor of the real `DistributedMatrix` +
  `DistTransport` versions of the same names re-exported from `distributed::prelude` (see
  `src/distributed/linalg/mod.rs`'s module doc).
- A bidirectional communication deadlock is eliminated by construction: `CommunicationChannel`
  used to hold one `Arc<Mutex<TcpStream>>` across every `.await`, so two ranks sending to each
  other simultaneously could each block waiting for the other's read to finish; the channel now
  uses `TcpStream::into_split` read/write halves so send and receive never contend on the same
  lock (`src/distributed/comm.rs`, regression-tested by
  `simultaneous_bidirectional_send_does_not_deadlock`).

**Carried over from the previous 0.4.1 pass**
- erfinv/erfcinv precision (Winitzki 2008 initial guess + 8 Halley iterations; error now < 1e-9).
- Bessel Y_n/K_n (DLMF series + upward recurrence, replacing formulas that divided by zero or used
  a dead branch for integer orders).
- Incomplete gamma and elliptic integrals: test tolerances tightened to ~1e-9/1e-10 against the
  already-correct series/AGM implementations.
- irfft/irfft2 Hermitian reconstruction (`mirror_start` off-by-one for even/odd `n`).
- `split` axis=1 re-enabled.
- Schur reconstruction assertion (`A = Q*T*Q^T`) re-enabled at 1e-8.
- Criterion/alloca linker failure on Apple Silicon, fixed by `patches/alloca/`.
- Gamma-distribution scale-parameter double-inversion workaround removed (upstream fixed).
- RNG seeding reproducibility: `Generator::random()` now seeds a per-call RNG from the
  bit-generator stream; 8 previously-`#[ignore]`d seeding tests re-enabled.
- GPU buffer-mapping errors now propagate as `Result` instead of `.expect()`-panicking (required
  by `wgpu` 30's fallible `get_mapped_range()`).
- 11 test files' RNG-race flakiness fixed by adding `#[serial]` around shared-global-state tests.

### Performance

Figures are release-mode measurements taken from the tables already checked into the modules that
own them (see BENCHMARKING.md for the min-over-alternating-A/B methodology); each row names its
source, and figures that could not be re-derived from a checked-in table say so instead of
restating an unverified multiplier as fact.

| Change | Measurement | Source |
|---|---|---|
| Matmul dispatch (`general_mat_mul`, row-split) | 2.51x @ 8^3 up to 18.82x @ 512^3 vs. the pre-existing loop; no regressed shape | `src/kernels/gemm.rs` module doc table |
| `sum`/`mean`/`var`/`min`/`max` kernel dispatch | 4.15x–6.97x at n=1,000,000 across three independent runs | `src/kernels/reduce.rs` module doc tables |
| `Some(axis)` stride hoist (`math::aggregation::sum`) | regression-tested at 64x4096 (`sum_axis_matches_naive_larger_2d`) | `src/math/aggregation.rs`; the wave's own report cites a much larger multiplier at this size, not re-derived here |
| COW mutation guard (`Arc::make_mut` uniqueness check) | a few ns of fixed absolute cost per `&mut` entry point on an already-unique array; bulk acquisition (`array_mut()`/`as_slice_mut()` once per call, not per element) keeps relative overhead under the bench's own <5% target | `bench/cow_mutation_guard.rs` module doc + `report_mutation_guard` |
| Bulk-acquisition hot loops (`matrix_decomp::{lu, pivoted_cholesky, qr::householder_qr}`) | ~2–5x faster at n=128/256 after converting a per-element `set()` loop to one bulk `array_mut()` | `bench/cow_mutation_guard.rs` module doc |
| Expression fusion (`.expr()...eval()` vs. eager) | 1.00x–1.20x @ n=1,000 up to 1.45x–1.69x @ n=1,000,000 | `src/expr/fused_eval.rs` module doc table |
| `Array::from_vec_shape` adoption (e.g. `Array::full`) | reported by the implementing wave; not independently re-benchmarked for this changelog | — |

### Known Upstream Issues

Documented in-tree with repro/tripwire tests so an upstream fix becomes visible when it lands:

- **`scirs2_core::simd_ops::SimdUnifiedOps::{simd_variance, simd_std}`** hardcode sample (n-1)
  variance and are unsuitable for NumPy (population) semantics. Never called from this crate's own
  reduction path; see the module docs in `src/stats/basic.rs` and `src/kernels/reduce.rs`.
- **`simd_min_element`/`simd_max_element`** return a wrong finite value for certain NaN
  placements — a live bug, not a NaN-convention difference. Tripwire test:
  `simd_max_element_upstream_wrong_value_is_a_live_bug_not_just_new_nan_convention` in
  `src/stats/basic.rs`, which calls the upstream function directly and pins the currently-wrong
  value.
- **`scirs2_fft::fftn`** has two confirmed normalization bugs (see Fixed, above); documented and
  worked around in `src/fft/numpy_parity.rs`.
- **`scirs2-core`'s `simd_matrix_multiply`** has a ~4x performance cliff whenever
  `k % 128 != 0`; documented in `src/kernels/gemm.rs` and `bench/matmul_dispatch_benchmark.rs`,
  and is part of why the default matmul tier no longer routes through it.

### Test Coverage

- `cargo nextest run --workspace` (default features): **5046 tests run, 5046 passed, 0 failed,
  17 skipped**.
- `cargo nextest run --workspace --features matrix_decomp,validation,unstable,fast,scirs,gpu,lapack,arrow,parquet,netcdf,matlab,messagepack,bson,io-all,wasm,distributed,visualization,ci-safe`
  (every feature except `python` — `pyo3`'s `extension-module` feature structurally cannot link
  into a `cargo test`/nextest binary; `scripts/ci-local.sh` checks `python` separately via
  `cargo check --features python` instead): **5635 tests run, 5635 passed, 0 failed, 31 skipped**.
- `cargo test --doc`: **842 passed, 0 failed, 88 ignored**.

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
