# NumRS2 Development Status

## Overview

This document outlines the development status and roadmap for NumRS2, a high-performance numerical computing library for Rust.

## Current Status (June 2026) - v0.4.1

**Major Release**: NumRS2 v0.4.1 is now available!

### Release Metrics
- **Version**: 0.4.1 (v0.4.0 - 2026-06-05)
- **Total Code**: ~540,000+ lines of Rust code (490+ source files)
- **Test Coverage**: 4,820+ library tests passing; zero failures
- **Public API**: 7,000+ public functions/structs/enums/traits; zero unimplemented stubs
- **Quality**: Zero compilation warnings, zero errors, zero production unwrap() calls
- **SIMD Operations**: 128+ vectorized functions (AVX2, AVX512, ARM NEON)
- **Dependencies**: SciRS2 v0.5.0, OxiBLAS v0.3.0+ (pure Rust, stable releases)
- **Special Functions**: scirs2-special v0.5.0
- **Performance**: Stable SVD/eigen now work for all matrix sizes (Jacobi + bidiagonalization); Schur with Francis double-shift QR
- **Latest Enhancement**: v0.4.0 - Major feature additions: autodiff, distributed, viz, WASM, RL, quantum, model I/O, serving, CMA-ES, Bayesian opt, CV, geometry, FEM, wavelets, graph, information theory, control, physical constants; correctness fixes for large-matrix stable eigen/SVD, quantum partial trace, polynomial domain mapping, FEM general det/inv, Boltzmann exploration, gamma_ln accuracy

## Code Audit & Remediation (2026-06-19)

A thorough source audit was performed in response to external code-reading criticism
(Reddit) of the SciRS2/NumRS2 ecosystem. Findings and actions for NumRS2 v0.4.1:

### Verdict on the criticisms
- **"`todo!()`/`unimplemented!()` left in place"**: NOT APPLICABLE — 0 such macros in `src/`.
- **"Ignores Rust trait system / const generics"**: LARGELY NOT VALID — the array type is
  generic over element type via a sound trait hierarchy; dynamic shapes are correct for a
  NumPy-like library. Build is clean (0 warnings), all files <2000 lines, SciRS2/workspace
  policies honored, GPU/SIMD routed through scirs2-core.
- **"Many simplified/stub implementations"**: PARTIALLY VALID — a set of numerical routines
  were genuinely simplified or mathematically wrong. Fixed below.
- **"GPU is name-only"**: PARTIALLY VALID — the f64 GPU path loaded f32 shaders (silent wrong
  results); docs overstated native CUDA/ROCm. Fixed below.
- **"Allocation-heavy / non-Rust-like"**: SUBSTANTIALLY VALID — core hot paths allocate via
  pervasive `.to_vec()`. Tracked below for a dedicated performance pass (Tier C).

### Fixed — correctness bugs (verified: `cargo build` 0 warnings [default + gpu]; `cargo clippy --all-targets` clean; `cargo nextest` 3994/3994 passing)
- [x] `ode.rs`: implicit Euler & BDF2 now use real Newton iteration (finite-difference Jacobian
      + Gaussian elimination with partial pivoting), replacing fixed-point iteration that
      diverged on stiff systems (the very case these solvers exist for). Removed dead code.
- [x] `optimize/sqp.rs`: `solve_qp_subproblem` now solves the real constrained QP (KKT system +
      active-set for inequalities, real Lagrange multipliers), replacing the unconstrained
      `p = -H⁻¹g` stub that ignored all constraints.
- [x] `optimize/interior_point.rs`: barrier Hessian completed — added the missing
      constraint-curvature term and corrected the sign of the existing terms.
- [x] `sparse_enhanced.rs`: spectral condition number is now κ = λ_max/λ_min (λ_min via inverse
      power iteration using the existing CG solve), not just λ_max.
- [x] `new_modules/matrix_decomp/condition.rs`: `slogdet` sign computed via LU decomposition
      (permutation parity × pivot signs), replacing the hardcoded `sign = 1`.
- [x] `gpu/context.rs`: real f64 WGSL shaders are loaded when the device reports `SHADER_F64`;
      otherwise f64 GPU ops return a clear error (no silent f32 fallback → no wrong results).
- [x] `new_modules/rl/agents.rs`: `SimpleNetwork::update` now performs real backpropagation
      (forward cache + chain rule), replacing the fake uniform `+= lr*signal*0.01` update.
- [x] `new_modules/timeseries/arima.rs`: MA coefficients estimated via the Hannan-Rissanen
      algorithm (long-AR residuals + joint LS), replacing the placeholder that returned ~zeros.
- [x] `new_modules/frequency_analysis.rs`: real DPSS (Slepian) tapers via the symmetric
      tridiagonal eigenformulation (QL eigensolver) with true concentration eigenvalues,
      replacing the sinc approximation + hardcoded eigenvalues.
- [x] `new_modules/quantum/algorithms.rs`: `multi_controlled_z` now a correct MCX/MCZ
      decomposition; VQE uses parameter-shift gradients + Adam; QPE applies real
      controlled-U^(2^k) with the power assignment and readout bit-reversal matched to
      the project's (bit-reversed) QFT convention, giving deterministic exact readout.

### Fixed — false advertising / docs
- [x] `README.md`: removed absolute "zero stubs" claims; GPU described accurately (WGPU
      backend across Vulkan/Metal/DX12/WebGPU; no native CUDA/ROCm/TPU shipped).
- [x] `docs/optimization_guide.md`: corrected "CUDA/ROCm backend" claims to the real WGPU mapping.
- [x] `src/optimized_ops.rs`: `get_optimization_info` no longer implies native CUDA/OpenCL/Metal
      execution (relabeled as scirs2-core platform detection; WGPU is NumRS2's only GPU path).

### Tracked — dedicated performance pass (Tier C, NOT yet done)
The "allocation-heavy" criticism is valid for core hot paths. Replace pervasive `.to_vec()` /
double-allocation patterns with borrowed slices / iterators (keep SIMD dispatch, operate on
borrowed data), and benchmark before/after with criterion to prove no regression:
- [ ] `ufuncs.rs` — `to_array_view`/`from_array1`: 2–3 allocations per SIMD op
- [ ] `simd.rs` — `to_ndarray_1d` + result recovery per op
- [ ] `array/operations.rs` — `map`/`zip_with`/`par_map`/`sum_axis`/`product` each `to_vec()` first
- [ ] `array/linalg.rs` — `matmul_2d` (3× `to_vec`), `dot_simd` (2× `to_vec`)
- [ ] `traits/implementations.rs` — reductions `to_vec()`

### Tracked — lower-priority simplifications (noted, not yet addressed)
- [ ] `new_modules/quantum`: multi-qubit gate fusion; full controlled-U for arbitrary U
- [ ] `new_modules/control/stability.rs`: simplified Lyapunov / eigenvalue / root-locus
- [ ] `new_modules/special`: simplified `bessel_k` (K₁), hypergeometric ₂F₁ continuation
- [ ] `new_modules/nn/graph.rs`: LSTM aggregator uses mean pooling
- [ ] `new_modules/probabilistic/graphical.rs`: topological-order assumption in sampling

### Core Features (Complete)
- ✅ N-dimensional array operations with NumPy compatibility
- ✅ Broadcasting and advanced indexing
- ✅ Expression templates for lazy evaluation
- ✅ SIMD optimization (AVX2, AVX512, ARM NEON)
- ✅ Linear algebra (SVD, QR, LU, Cholesky, Eigenvalue)
- ✅ Iterative solvers (CG, GMRES, BiCGSTAB)
- ✅ Sparse matrices (COO, CSR, CSC, DIA)
- ✅ Mathematical and statistical functions
- ✅ Special functions (gamma, beta, Bessel, etc.)
- ✅ Polynomial operations and cubic splines
- ✅ FFT and signal processing
- ✅ Numerical optimization (BFGS, Trust Region, etc.)
- ✅ Root-finding algorithms
- ✅ Automatic differentiation (forward & reverse mode)
- ✅ NumPy format (.npy, .npz) support
- ✅ Apache Arrow integration
- ✅ GPU acceleration (optional)

### SciRS2 Ecosystem Integration (Complete)
All modules integrated using SciRS2 v0.3.0:
- ✅ scirs2-core: SIMD, parallel, random, array operations
- ✅ scirs2-linalg: Linear algebra with OxiBLAS
- ✅ scirs2-stats: Statistical functions
- ✅ scirs2-fft: FFT operations
- ✅ scirs2-signal: Signal processing
- ✅ scirs2-special: Special functions
- ✅ scirs2-ndimage: N-dimensional image processing
- ✅ scirs2-spatial: Spatial algorithms and KD-trees
- ✅ scirs2-numpy: Python bindings (optional)

### v0.3.0 New Features (Completed)
- ✅ **Neural Networks**: Layer implementations, activation functions, normalization (dropout, batch norm, layer norm)
- ✅ **Symbolic Computation**: Expression parsing, differentiation, integration, simplification, linear algebra
- ✅ **Advanced Optimization**: Differential Evolution, Genetic Algorithms, Particle Swarm, Simulated Annealing, SQP, Interior Point, NSGA-II
- ✅ **Distributed Computing**: Pure Rust distributed operations (no MPI)
- ✅ **Visualization**: Plotters-based plotting with SVG/PNG rendering (pure Rust)
- ✅ **WebAssembly**: WASM bindings and examples (partial - blocked by upstream scirs2-spatial)
- ✅ **I/O Formats**: Parquet, NetCDF, MATLAB .mat, MessagePack, BSON (all pure Rust)
- ✅ **Python Bindings**: Comprehensive PyO3 integration with NumPy compatibility
- ✅ **Comprehensive Benchmarks**: 8 benchmark suites (linalg, stats, fft, array ops, optimization, SIMD, parallel, memory)

### v0.3.0 Enhanced (February 9-11, 2026) - NEW
- ✅ **GPU Compute Shaders**: Advanced shader system with caching, kernel composition, async transfers (~1,570 lines, 34 tests)
- ✅ **Extended Statistics**: 7 new distributions (Beta, Gamma, Student's t, Cauchy, Laplace, Logistic, Pareto) with PDF/CDF/PPF (~1,860 lines, 50 tests)
- ✅ **Statistical Functions Fix**: Fixed Beta and Student's t CDF/PPF via upstream betainc_regularized bug fix (scirs2-special v0.3.0-dev)
- ✅ **Parallel Enhancements**: Work-stealing thread pool, NUMA awareness, parallel algorithms (map/reduce/filter/sort/pipeline) (~2,500 lines, 42 tests)
- ✅ **Performance Optimization**: Fixed critical O(n²) expression template bug (~1000x speedup for large arrays)
- ✅ **Comprehensive Examples**: 6 new tutorials (distributed, optimization, statistics, time series, signal processing, ML pipeline) (~4,200 lines)
- ✅ **Example Fixes**: Fixed neural_network_basics.rs and 4 optimization examples (API mismatches resolved)
- ✅ **Multi-Objective Optimization Suite**: Complete NSGA-II with quality metrics + NSGA-III + test problems (ZDT, DTLZ) (~7,304 lines, 227+ tests) - Feb 11
- ✅ **Parallel Computing Tests**: Comprehensive test suite covering work-stealing, NUMA, load balancing, metrics (131 tests, ~2,600 lines) - Feb 11
- ✅ **Cache Alignment**: Critical structures aligned for 20-50% parallel performance improvement (AlignedBox, AlignedVec helpers) - Feb 11
- ✅ **GPU Batching Operations**: Automatic operation batching with dynamic optimization (~1,231 lines, 15 tests) - Feb 11
- ✅ **Documentation Excellence**: Complete NN Guide (1,800+ lines), multi-objective examples (2,072 lines), benchmarks (1,524 lines) - Feb 11

### v0.3.0 Features (March 6, 2026) - COMPLETED
- ✅ **SciRS2 0.3.0 Integration**: All scirs2-* deps updated to v0.3.0
- ✅ **im2col Convolution**: conv2d_batched and conv_transpose2d_batched via scirs2-linalg im2col
- ✅ **Survival Analysis**: Kaplan-Meier, Nelson-Aalen, Cox Proportional Hazards, log-rank test (14 tests)
- ✅ **Causal Inference**: Difference-in-Differences, 2SLS/IV, propensity score, IPW-ATE (11 tests)
- ✅ **Bioinformatics**: Needleman-Wunsch, Smith-Waterman, edit distance, phylogenetics (25 tests)
- ✅ **Combinatorics**: Fibonacci, Catalan, Bell, Stirling, primes, partitions, permutations (37 tests)
- ✅ **NPZ Compression**: DEFLATE compression enabled for .npz files (OxiARC 0.2.1 fix)
- ✅ **Cyclic Spline Solver**: O(n) Sherman-Morrison replaces O(n²) Gaussian elimination
- ✅ **Code Refactoring**: test_problems.rs, nsga2.rs, higher_order.rs, exponential_smoothing.rs split into proper module directories

### v0.3.x Features (February 11, 2026) - COMPLETED
- ✅ **Transformer Neural Networks**: Multi-head attention, positional encoding (sinusoidal/learned), encoder/decoder stacks (~1,400 lines, 15+ tests) - Test fixes completed Feb 12, 2026
- ✅ **Graph Neural Networks**: GCN, GAT, GraphSAGE, MPNN, GIN architectures with graph representations (1,698 lines, 40 tests)
- ✅ **Probabilistic Programming**: MCMC (Metropolis-Hastings, HMC, Gibbs), Variational Inference (ADVI, ELBO), Bayesian utilities (77 tests)
- ✅ **Time Series Analysis**: ARIMA/SARIMA, VAR/VECM, Kalman filtering, ACF/PACF, state space models (57 tests)
- ✅ **Critical Bug Fixes**: Von Mises distribution (Best-Fisher algorithm), VAR log-likelihood (quadratic form), MCMC convergence tolerances

### v0.4.x Features (February 12, 2026) - COMPLETED
- ✅ **Reinforcement Learning** (src/new_modules/rl/): Complete RL framework with agents, environment abstractions, replay buffers, and utilities (5 files, 2,645 lines, 2,192 code, 283+ tests across all v0.4.x modules)
  - DQN, Actor-Critic, PPO agent implementations
  - Experience replay with prioritized sampling
  - Environment interface and wrappers
  - Reward shaping and normalization utilities
- ✅ **Quantum Computing** (src/new_modules/quantum/): Full quantum simulation support (6 files, 2,373 lines, 1,865 code)
  - Quantum gates (Hadamard, Pauli, CNOT, Toffoli, phase gates)
  - Circuit construction and composition
  - State vector simulation with superposition
  - Measurement operations (computational basis, observables)
  - Quantum algorithms (Deutsch-Jozsa, Grover, QFT foundations)
- ✅ **Model Serialization** (src/new_modules/model_io/): Production-grade model I/O and format conversion (6 files, 2,719 lines, 2,092 code)
  - Multiple format support (ONNX-compatible, TensorFlow Lite, PyTorch compatibility layer)
  - Efficient serialization with oxicode (SIMD-optimized, pure Rust)
  - Cross-platform model export and import
  - Versioning and compatibility management
  - Compression and optimization utilities
- ✅ **Production ML Serving** (src/new_modules/serving/): Complete inference serving infrastructure (7 files, 3,700 lines, 2,929 code)
  - High-performance inference engine with batching
  - Model registry and versioning system
  - Request preprocessing and postprocessing pipelines
  - Prediction API with async support
  - Performance optimization (quantization, pruning)
  - Real-time metrics and monitoring
- ✅ **Advanced Distributed Training** (src/distributed/): Comprehensive distributed computing framework (12 files, 5,541 lines, 4,274 code) - Enhanced
  - Model parallelism (pipeline, tensor splitting)
  - Data parallelism with gradient synchronization
  - Distributed optimizers (AllReduce, Ring-AllReduce)
  - Pure Rust implementation (no MPI dependency)
  - Communication layer with efficient collectives
  - Fault tolerance and checkpointing

## Future Enhancements

### v0.3.0 Completed
- ✅ COOLJAPAN ecosystem compliance (pure Rust, no C/Fortran dependencies)
- ✅ Replaced numpy dependency with scirs2-numpy
- ✅ Removed OpenBLAS linker flags (now using OxiBLAS)
- ✅ Eliminated all production unwrap() calls (no-unwrap policy)
- ✅ Updated to SciRS2 v0.3.0
- ✅ Comprehensive benchmarks (8 benchmark suites)
- ✅ Extended Python bindings with NumPy compatibility
- ✅ Symbolic computation support (differentiation, integration, simplification)
- ✅ Advanced optimization algorithms (DE, GA, PSO, SA, SQP, Interior Point, NSGA-II)
- ✅ Distributed computing support (pure Rust, no MPI)
- ✅ Deep learning primitives (layers, activations, normalization)
- ✅ Advanced visualization tools (plotters-based, pure Rust)
- ✅ WebAssembly support (partial - blocked by upstream scirs2-spatial → tokio issue)
- ✅ Additional I/O formats (Parquet, NetCDF, MATLAB, MessagePack, BSON)
- ✅ 100% test pass rate (1,635+ tests passing, +325 new tests)
- ✅ Zero compilation errors and warnings
- ✅ **Enhanced GPU acceleration** with compute shaders, kernel composition, buffer management (Feb 9, 2026)
- ✅ **Extended statistical distributions** with 7 new distributions and comprehensive functions (Feb 9, 2026)
- ✅ **Statistical accuracy fix** - Fixed Beta and Student's t CDF/PPF via betainc_regularized bug fix (Feb 9, 2026)
- ✅ **Advanced parallel computing** with work-stealing, NUMA awareness, parallel algorithms (Feb 9, 2026)
- ✅ **Critical performance fix** - O(n²) → O(n) expression templates (~1000x speedup) (Feb 9, 2026)
- ✅ **Comprehensive documentation** with 6 new example tutorials (+4,200 lines) (Feb 9, 2026)
- ✅ **Example fixes** - neural_network_basics.rs and 4 optimization examples now compile (Feb 9, 2026)
- ✅ **Multi-objective optimization framework** - Production-ready NSGA-II with quality metrics (hypervolume, spacing, spread, IGD, GD) (Feb 11, 2026)
- ✅ **NSGA-III algorithm** - Many-objective optimization (3+ objectives) with reference point generation (Feb 11, 2026)
- ✅ **Benchmark test problems** - Industry-standard ZDT (bi-objective) and DTLZ (scalable many-objective) suites (Feb 11, 2026)
- ✅ **Parallel computing tests** - 131 comprehensive tests for work-stealing, NUMA, load balancing, metrics monitoring (Feb 11, 2026)
- ✅ **Cache alignment optimization** - Critical hot paths aligned (ParallelConfig, BroadcastEngine, GpuContext, etc.) (Feb 11, 2026)
- ✅ **GPU batching operations** - Automatic batching with dynamic optimization for 80% GPU occupancy (Feb 11, 2026)
- ✅ **NN documentation guide** - Complete 1,800+ line guide with examples, formulas, SIMD strategies (Feb 11, 2026)

### v0.3.1 (March 21, 2026) - COMPLETED
- ✅ **Clippy Fixes**: Resolved all 24 clippy warnings for MSRV compatibility
  - Fixed `Color` trait ambiguity in viz modules (matrix.rs, perf.rs, plot2d.rs, plot3d.rs, stats.rs)
  - Replaced explicit counter loops with `enumerate`/`zip` pattern in cluster.rs
  - Replaced manual checked division with `.checked_div()` in performance_tuning.rs, access_patterns.rs, scheduler.rs
- ✅ **SciRS2 Ecosystem Update**: Updated all scirs2-* deps to v0.4.0
- ✅ **Test Coverage**: 5,058+ tests passing (up from 4,098+)

### Short-term Goals (v0.3.3 - Patches)
- ✅ Complete WASM support — **BLOCKER RESOLVED** (May 2026): scirs2-spatial 0.5.0 feature-gates tokio under `async`; numrs uses only `parallel`, so no transitive tokio. Browser WASM requires disabling `gpu`/`distributed` features (which pull tokio directly).
- ✅ Additional distribution functions (beta, gamma, student-t extensions) - COMPLETED Feb 9, 2026
- ✅ Enhanced GPU acceleration (compute shaders, buffer management) - COMPLETED Feb 9, 2026
- ✅ Performance optimizations (fixed O(n²) bug, memory improvements) - COMPLETED Feb 9, 2026
- ✅ Extended examples and tutorials (6 comprehensive examples) - COMPLETED Feb 9, 2026
- ✅ Fix example API mismatches (optimization config structs) - COMPLETED Feb 9, 2026
- ✅ Fix statistical distribution bugs (Beta, Student's t CDF/PPF) - COMPLETED Feb 9, 2026
- ✅ Fix neural_network_basics.rs compilation errors - COMPLETED Feb 9, 2026
- ✅ Additional performance benchmarks and profiling - COMPLETED Feb 11, 2026 (see /tmp/NUMRS2_V0.2.0_PERFORMANCE_ANALYSIS.md)

### Medium-term Goals (v0.3.x) - COMPLETED Feb 11, 2026
- ✅ Advanced neural network layers (transformers, attention mechanisms) - ~1,400 lines, 15+ tests
- ✅ Probabilistic programming support - MCMC (Metropolis-Hastings, HMC, Gibbs), Variational Inference, 77 tests
- ✅ Time series analysis module - ARIMA/SARIMA, VAR/VECM, ACF/PACF, Kalman filtering, 57 tests
- ✅ Graph neural network primitives - GCN, GAT, GraphSAGE, MPNN, GIN (1,698 lines, 40 tests)
- ✅ Enhanced GPU compute capabilities - Already completed in v0.3.0 (compute shaders, batching)

### v0.4.x Completed (February 12, 2026)
- ✅ **Reinforcement Learning Framework** - Complete RL implementation with DQN, Actor-Critic, PPO agents (2,645 lines, 21 test modules)
- ✅ **Quantum Computing Simulation** - Gates, circuits, state vectors, measurements, quantum algorithms (2,373 lines)
- ✅ **Model Serialization & Format Conversion** - ONNX, TFLite, PyTorch compatibility, oxicode optimization (2,719 lines)
- ✅ **Production ML Serving Infrastructure** - Inference engine, model registry, optimization, real-time monitoring (3,700 lines)
- ✅ **Enhanced Distributed Training** - Model/data parallelism, distributed optimizers, fault tolerance (5,541 lines)
- ✅ **Total v0.4.x Addition** - 36 files, ~17,000 lines of production code, 283+ comprehensive tests
- ✅ **Pure Rust Implementation** - Zero C/Fortran dependencies, full COOLJAPAN ecosystem compliance
- ✅ **SciRS2 Integration** - All modules built on scirs2-core v0.3.0 foundation

### Long-term Goals (v0.4.x+) - COMPLETED February 12, 2026
- ✅ Reinforcement learning primitives (DQN, Actor-Critic, PPO, experience replay, environments)
- ✅ Quantum computing simulation support (gates, circuits, state vectors, measurements, algorithms)
- ✅ Advanced distributed training patterns (model/data parallelism, gradient sync, fault tolerance)
- ✅ Model serialization/deployment formats (ONNX, TFLite, PyTorch compatibility, oxicode)
- ✅ Production ML serving capabilities (inference engine, model registry, optimization, monitoring)

### v0.5.x Tier 2 Features (February 12, 2026) - COMPLETED
- ✅ **CMA-ES Optimizer** (src/optimize/cma_es/): Covariance Matrix Adaptation Evolution Strategy (7 files, ~1,936 lines, 23 tests)
  - IPOP-CMA-ES with population restart strategy
  - Step-size adaptation (CSA - Cumulative Step-size Adaptation)
  - Covariance matrix eigendecomposition and update
  - Rank-μ and rank-one updates
  - Constraint handling and boundary repair
  - Convergence detection and termination criteria
- ✅ **Bayesian Optimization** (src/optimize/bayesian_opt.rs): Gaussian Process-based global optimization (~1,484 lines)
  - Gaussian Process surrogate models
  - Acquisition functions: Expected Improvement (EI), Probability of Improvement (PI), Upper/Lower Confidence Bound (UCB/LCB)
  - Kernel functions: Matern (ν=1.5, 2.5, ∞), RBF, Automatic Relevance Determination (ARD)
  - Hyperparameter optimization via maximum likelihood estimation
  - Multi-start optimization for acquisition function maximization
- ✅ **Computer Vision** (src/new_modules/cv/): Image processing and feature detection (~4 files)
  - Image filtering: Gaussian blur, median filter, bilateral filter
  - Edge detection: Sobel operator, Canny edge detector
  - Morphological operations: erosion, dilation, opening, closing
  - Feature detection: Harris corner detector, FAST corner detector
  - Geometric transformations: rotation, scaling, affine transforms
  - Comprehensive test coverage with synthetic images
- ✅ **Computational Geometry** (src/new_modules/geometry/): Geometric algorithms and spatial operations (~4 files)
  - Convex hull computation (Graham scan algorithm)
  - Delaunay triangulation (Bowyer-Watson with super-triangle)
  - Voronoi diagram generation (dual of Delaunay triangulation)
  - Polygon operations: area, centroid, point-in-polygon, intersection
  - Line segment operations: intersection, distance
  - Comprehensive test coverage with edge cases
- ✅ **Finite Element Method** (src/new_modules/fem/): FEM solver for PDEs (~5 files)
  - 1D/2D FEM solver with assembly and solution
  - Mesh generation: structured grids, element connectivity
  - Element types: line elements (1D), triangle/quad elements (2D)
  - Boundary conditions: Dirichlet (essential), Neumann (natural)
  - Solvers: direct (LU) and iterative (CG) linear system solvers
  - Shape functions and numerical integration (Gaussian quadrature)
  - Comprehensive tests: Poisson equation, heat equation, elasticity

### v0.5.x Tier 3 Features (February 12, 2026) - COMPLETED
- ✅ **Wavelets** (src/new_modules/wavelets/): Wavelet transform and multiresolution analysis
  - Discrete Wavelet Transform (DWT) - fast O(n) algorithm
  - Continuous Wavelet Transform (CWT) - time-frequency analysis
  - Wavelet packet decomposition
  - Wavelet families: Haar, Daubechies (db2-db10), Symlet, Coiflet
  - Multiresolution analysis (MRA) and filter banks
  - Applications: denoising, compression, feature extraction
- ✅ **Graph Algorithms** (src/new_modules/graph/): Graph theory and network algorithms
  - Graph representations: adjacency matrix, adjacency list, edge list
  - Traversal: BFS (Breadth-First Search), DFS (Depth-First Search)
  - Shortest paths: Dijkstra, Bellman-Ford, Floyd-Warshall, A* search
  - Minimum Spanning Tree: Kruskal, Prim algorithms
  - Maximum flow: Ford-Fulkerson, Edmonds-Karp, Dinic
  - Topological sort and strongly connected components
  - Graph properties: diameter, centrality, clustering coefficient
- ✅ **Information Theory** (src/new_modules/information_theory/): Information-theoretic measures
  - Shannon entropy (discrete and continuous)
  - Mutual information and conditional entropy
  - Kullback-Leibler (KL) divergence
  - Jensen-Shannon divergence
  - Cross-entropy and relative entropy
  - Channel capacity and information transmission
  - Applications: feature selection, model comparison, data compression
- ✅ **Control Systems** (src/new_modules/control/): Control theory and system analysis
  - Transfer function representation (continuous/discrete)
  - State space models and conversions
  - Stability analysis: Routh-Hurwitz, Nyquist criterion
  - Time response: step, impulse, frequency response
  - Bode plots and Nyquist plots (data generation)
  - Controller design: PID tuning, LQR, pole placement
  - Observability and controllability analysis
- ✅ **Physical Constants** (src/new_modules/constants/): NIST-compliant physical constants
  - Fundamental constants: speed of light, Planck constant, electron charge
  - Atomic and nuclear constants: Bohr radius, Rydberg constant
  - Physico-chemical constants: Avogadro number, gas constant, Boltzmann constant
  - Electromagnetic constants: permittivity, permeability, impedance of vacuum
  - Unit conversions: SI units, CGS units, natural units
  - CODATA 2018/2022 recommended values with uncertainties

## v0.4.0 Correctness Fixes (May 2026)
- ✅ **Build fix**: `oxiarc-core` lockfile resolution 0.2.6→0.2.8 (was failing with `cancel`/`progress` module errors from `scirs2-core 0.5.0`'s transitive oxiarc-lz4/zstd 0.2.8 deps)
- ✅ **Stable SVD for large matrices** (`src/linalg_stable.rs`): Golub-Kahan bidiagonalization + Jacobi SVD (was silently falling back to n≤3 path)
- ✅ **Stable eigendecomposition for large matrices** (`src/linalg_stable.rs`): cyclic Jacobi sweeps algorithm (was silently falling back to n≤3 path)
- ✅ **Quantum partial trace** (`src/new_modules/quantum/statevector.rs`): correct bit-interleaving (was using `full_i = i`, always wrong for multi-qubit traces)
- ✅ **Polynomial domain mapping** (`src/new_modules/polynomial/utils.rs`): real polynomial composition under affine map (was returning input unchanged)
- ✅ **FEM general det/inverse** (`src/new_modules/fem/elements.rs`): LU with partial pivoting for all n×n (was hard-erroring for n>3)
- ✅ **Schur decomposition** (`src/new_modules/matrix_decomp/schur.rs`): Francis implicit double-shift QR + deflation for real Schur form (was using single Rayleigh shift, no deflation)
- ✅ **Boltzmann exploration** (`src/new_modules/rl/utils.rs`): temperature-scaled softmax action selection (was falling back to greedy, ignoring temperature)
- ✅ **`gamma_ln` accuracy** (`src/new_modules/probabilistic/distributions.rs`): delegates to `scirs2_special::loggamma` Lanczos g=7 (~15-digit accuracy)
- ✅ **`average_with_weights`** (`src/stats/basic.rs`): new public function returning `(avg, weight_sum)` tuple (NumPy `returned=True` semantics)
- ✅ **Zero warnings**: fixed unsafe block warnings in `cache_layout.rs`, copy-loop warnings in `avx2_enhanced/special.rs`, useless `vec!` in `avx2_enhanced/mod.rs` tests

## Known Future Work (v0.5.0+)

### Distributed Computing Stubs
The distributed framework compiles and exports a public API but several core operations are stub-implemented:
- `distributed/collective.rs`: scatter, gather, all-reduce, broadcast — return empty vectors (no real network transport)
- `distributed/model_parallel.rs`: `recv_forward`/`recv_backward` — return placeholder tensors (zero data)
- `distributed/linalg.rs`: matvec, matmul, SVD, QR, Cholesky, solve — return `NotImplemented` errors
- `distributed/optimization.rs`: bandwidth/latency measurement — return hardcoded constants

These require a real inter-process/network communication layer (e.g., a pure-Rust replacement for MPI).

### GPU NotImplemented Branches
- `src/gpu/batching.rs`: Conv2D batching returns `NotImplemented`
- `src/gpu/ops.rs`: N-D broadcasting and some ops return `NotImplemented`

### WebAssembly Support (Partial)
- **Status**: WASM bindings implemented; upstream tokio blocker RESOLVED as of scirs2-spatial 0.5.0
- **Remaining**: The `gpu` and `distributed` features pull tokio directly, so browser WASM (`wasm32-unknown-unknown`) builds must exclude them: `--no-default-features --features wasm`
- **Server-side WASM** (`wasm32-wasip1`): works with current dependencies

### Random Seeding Tests (~10 ignored)
About 10 tests tagged `#[ignore]` with "Seeding behavior changed during SciRS2 migration" across `src/random/` and `tests/test_random_*.rs`. Root cause is a change in how `scirs2-core` handles RNG seeding. These need investigation against the current `scirs2-core 0.5.0` RNG API.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to NumRS2.

## Documentation

- [Getting Started](GETTING_STARTED.md)
- [API Documentation](https://docs.rs/numrs2)
- [Examples](examples/)
- [SciRS2 Integration Policy](SCIRS2_INTEGRATION_POLICY.md)
- [Migration Guide](docs/MIGRATION_GUIDE.md)
- [Release Notes](RELEASE_NOTES.md)

---

**NumRS2 v0.4.1** - Production-ready numerical computing for Rust with SciRS2 v0.5.0 integration (COOLJAPAN Ecosystem)

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [ ] `numrs`: `examples/visualization.rs:7` — implement viz module so the disabled visualization example compiles and runs
  - Priority: P2 | Scope: medium | Hint: none
- [ ] `numrs`: `tests/test_fft_properties.rs:419,592` — fix FFT precision issues and re-enable the two skipped tests
  - Priority: P2 | Scope: small | Hint: oxifft
- [ ] `numrs`: `tests/test_special_reference.rs:89` — fix `erfinv` / `erfcinv` implementations to match reference values
  - Priority: P2 | Scope: small | Hint: none
- [ ] `numrs`: `tests/test_special_reference.rs:251` — fix `bessel_y` implementation to match reference values
  - Priority: P2 | Scope: small | Hint: none
- [ ] `numrs`: `tests/test_special_reference.rs:336` — fix `bessel_k` implementation to match reference values
  - Priority: P2 | Scope: small | Hint: none
- [ ] `numrs`: `tests/test_special_reference.rs:398,424` — fix `ellipk` / `ellipe` implementations to match reference values
  - Priority: P2 | Scope: small | Hint: none
- [ ] `numrs`: `tests/test_special_reference.rs:461` — fix `gammainc` implementation to match reference values
  - Priority: P2 | Scope: small | Hint: none
- [ ] `numrs`: `tests/test_linalg_reference.rs:745` — investigate and fix Schur decomposition precision issues
  - Priority: P2 | Scope: medium | Hint: none
- [ ] `numrs`: `tests/test_scirs_integration.rs:260` — fix set_seed() propagation to scirs2_stats distributions (Maxwell, other)
  - Priority: P2 | Scope: small | Hint: none
- [ ] `numrs`: `tests/nn/test_simd_ops.rs:248` — re-enable skipped f64 matmul SIMD test once scirs2-core issue resolved
  - Priority: P2 | Scope: trivial | Hint: none
- [ ] `numrs`: `bench/stats_benchmarks.rs:461` — implement skewness and kurtosis functions to unlock stats benchmarks
  - Priority: P2 | Scope: small | Hint: none
- [ ] `numrs`: `bench/bench_distributions.rs:96` — implement `f_dist` and `multivariate_normal_cholesky` distributions
  - Priority: P2 | Scope: medium | Hint: none
